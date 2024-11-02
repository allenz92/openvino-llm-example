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

def convert_to_fp16():
    if (fp16_model_dir / "openvino_model.xml").exists():
        return
    remote_code = llm_model_configuration.get("remote_code", False)
    export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format fp16".format(pt_model_id)
    if remote_code:
        export_command_base += " --trust-remote-code"
    export_command = export_command_base + " " + str(fp16_model_dir)
    print(export_command)
    system(export_command)


def convert_to_int8():
    if (int8_model_dir / "openvino_model.xml").exists():
        return
    int8_model_dir.mkdir(parents=True, exist_ok=True)
    remote_code = llm_model_configuration.get("remote_code", False)
    export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format int8".format(pt_model_id)
    if remote_code:
        export_command_base += " --trust-remote-code"
    export_command = export_command_base + " " + str(int8_model_dir)
    print(export_command)
    system(export_command)


def convert_to_int4():
    compression_configs = {
        "zephyr-7b-beta": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "mistral-7b": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "minicpm-2b-dpo": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "gemma-2b-it": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "notus-7b-v1": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "neural-chat-7b-v3-1": {
            "sym": True,
            "group_size": 64,
            "ratio": 0.6,
        },
        "llama-2-chat-7b": {
            "sym": True,
            "group_size": 128,
            "ratio": 0.8,
        },
        "llama-3-8b-instruct": {
            "sym": True,
            "group_size": 128,
            "ratio": 0.8,
        },
        "gemma-7b-it": {
            "sym": True,
            "group_size": 128,
            "ratio": 0.8,
        },
        "chatglm2-6b": {
            "sym": True,
            "group_size": 128,
            "ratio": 0.72,
        },
        "qwen-7b-chat": {"sym": True, "group_size": 128, "ratio": 0.6},
        "red-pajama-3b-chat": {
            "sym": False,
            "group_size": 128,
            "ratio": 0.5,
        },
        "qwen2.5-7b-instruct": {"sym": True, "group_size": 128, "ratio": 1.0},
        "qwen2.5-3b-instruct": {"sym": True, "group_size": 128, "ratio": 1.0},
        "qwen2.5-14b-instruct": {"sym": True, "group_size": 128, "ratio": 1.0},
        "qwen2.5-1.5b-instruct": {"sym": True, "group_size": 128, "ratio": 1.0},
        "qwen2.5-0.5b-instruct": {"sym": True, "group_size": 128, "ratio": 1.0},
        "default": {
            "sym": False,
            "group_size": 128,
            "ratio": 0.8,
        },
    }

    model_compression_params = compression_configs.get(llm_model_id, compression_configs["default"])
    if (int4_model_dir / "openvino_model.xml").exists():
        return
    remote_code = llm_model_configuration.get("remote_code", False)
    export_command_base = "optimum-cli export openvino --model {} --task text-generation-with-past --weight-format int4".format(pt_model_id)
    int4_compression_args = " --group-size {} --ratio {}".format(model_compression_params["group_size"], model_compression_params["ratio"])
    if model_compression_params["sym"]:
        int4_compression_args += " --sym"
    if enable_awq:
        int4_compression_args += " --awq --dataset wikitext2 --num-samples 128"
    export_command_base += int4_compression_args
    if remote_code:
        export_command_base += " --trust-remote-code"
    export_command = export_command_base + " " + str(int4_model_dir)
    print(export_command)
    system(export_command)


# compression
if prepare_fp16_model:
    convert_to_fp16()
if prepare_int8_model:
    convert_to_int8()
if prepare_int4_model:
    convert_to_int4()


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

for query in querys:
    streaming_response = query_engine.query(query)
    streaming_response.print_response_stream()
