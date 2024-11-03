# -*- coding: utf-8 -*-
from pathlib import Path
from notebook_utils import optimize_bge_embedding
import os
import shutil

from prepare_embedding_model import get_embedding_model
from prepare_llm_model import get_llm_model
from prepare_vector_store import get_vector_store

system = os.system

from common import *

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"


from prepare_rerank_model import gen_rerank_model

reranker = gen_rerank_model(rerank_model_id, rerank_device, 2)


from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.readers.file import PyMuPDFReader, FlatReader
from llama_index.vector_stores.faiss import FaissVectorStore
from transformers import StoppingCriteria, StoppingCriteriaList
import faiss
import torch

if model_language == "English":
    text_example_path = "text_example_en.pdf"
else:
    text_example_path = "doc_emb/text1.txt"


class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


llm = get_llm_model(llm_model_id, model_dir, llm_model_configuration, llm_device)

stop_tokens = llm_model_configuration.get("stop_tokens")

if stop_tokens is not None:
    if isinstance(stop_tokens[0], str):
        stop_tokens = llm._tokenizer.convert_tokens_to_ids(stop_tokens)
    stop_tokens = [StopOnTokens(stop_tokens)]


embedding = get_embedding_model(embedding_model_id, USING_NPU)
# dimensions of embedding model
d = embedding._model.request.outputs[0].get_partial_shape()[2].get_length()
print(d)
faiss_index = faiss.IndexFlatL2(d)
Settings.embed_model = embedding

llm.max_new_tokens = 2048
if stop_tokens is not None:
    llm._stopping_criteria = StoppingCriteriaList(stop_tokens)
Settings.llm = llm

index_path = "vector-store"

index = get_vector_store(index_path)

query_engine = index.as_query_engine(streaming=True, similarity_top_k=3, node_postprocessors=[reranker])

querys = ["石头G20S在什么时候上市的？",
"预算有限，可以买哪个扫地机器人？" ]

import gradio as gr

def predict(message, history):
    response = query_engine.query(message).response_gen
    buf = ''
    for text in response:
        buf += text
        yield buf

gr.ChatInterface(predict, type="messages").launch()
