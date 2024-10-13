import os
import logging
import ollama
import numpy as np
from glob import glob
import json
from rich import print
from typing import List

from openai import AsyncOpenAI
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

# Assumed llm model settings
LLM_BASE_URL = "https://api.deepseek.com"
LLM_API_KEY = "sk-05efb5ad75d84d9399e51bb5c2a46ff8"
MODEL = "deepseek-chat"

# Assumed embedding model settings
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_MODEL_DIM = 768
EMBEDDING_MODEL_MAX_TOKENS = 8192


async def llm_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(
        api_key=LLM_API_KEY, base_url=LLM_BASE_URL
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    response = await openai_async_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


def query(working_dir: str, content: str, mode: str = "local"):
    
    rag = GraphRAG(
        working_dir=working_dir,
        best_model_func=llm_model_if_cache,
        cheap_model_func=llm_model_if_cache,
        embedding_func=ollama_embedding,
    )
    print(
        rag.query(
            content, param=QueryParam(mode=mode)
        )
    )


def insert(working_dir: str, doc_list: List[str]):
    from time import time

    remove_if_exist(f"{working_dir}/vdb_entities.json")
    remove_if_exist(f"{working_dir}/kv_store_full_docs.json")
    remove_if_exist(f"{working_dir}/kv_store_text_chunks.json")
    remove_if_exist(f"{working_dir}/kv_store_community_reports.json")
    remove_if_exist(f"{working_dir}/graph_chunk_entity_relation.graphml")

    rag = GraphRAG(
        working_dir=working_dir,
        enable_llm_cache=True,
        best_model_func=llm_model_if_cache,
        cheap_model_func=llm_model_if_cache,
        embedding_func=ollama_embedding,
    )
    start = time()
    rag.insert(doc_list)
    print("indexing time:", time() - start)


# We're using Ollama to generate embeddings for the BGE model
@wrap_embedding_func_with_attrs(
    embedding_dim=EMBEDDING_MODEL_DIM,
    max_token_size=EMBEDDING_MODEL_MAX_TOKENS,
)

async def ollama_embedding(texts :list[str]) -> np.ndarray:
    embed_text = []
    for text in texts:
      data = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
      embed_text.append(data["embedding"])
    
    return embed_text

def is_empty_args(args: dict):
    return len(args) == 0

def main():
    with open("insert_and_query.json", "r") as f:
        args = json.load(f)

    WORKING_DIR = args["working_dir"]
    if not is_empty_args(args["insert"]):
        doc_list = []
        doc_dir = args["insert"]["doc_dir"]
        doc_file_list = glob(doc_dir + "/*.txt")
        for file in doc_file_list:
            print(f"Reading {file}...")
            with open(file, encoding="utf-8-sig") as f:
                doc_list.append(f.read())
        insert(WORKING_DIR, doc_list)
    if not is_empty_args(args["query"]):
        query(
            WORKING_DIR,
            content=args["query"]["content"],
            mode=args["query"]["mode"]
        )

if __name__ == "__main__":
    main()
