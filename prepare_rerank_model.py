import shutil
import os
from pathlib import Path

system = os.system
from llm_config import (
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_RERANK_MODELS,
    SUPPORTED_LLM_MODELS,
)

from llama_index.postprocessor.openvino_rerank import OpenVINORerank


os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
# 参数设置
model_language = 'Chinese'
llm_model_id = 'qwen2.5-1.5b-instruct'
enable_awq = 1
embedding_model_id = 'bge-small-zh-v1.5'
rerank_model_id = 'bge-reranker-v2-m3'
embedding_device = 'CPU'
rerank_device = 'CPU'
llm_device = 'CPU'


def gen_rerank_model(rerank_model_id, rerank_device, top_n=2):
    reranker = OpenVINORerank(model_id_or_path=rerank_model_id, device=rerank_device, top_n=top_n)
    return reranker

if __name__ == '__main__':

    rerank_model_configuration = SUPPORTED_RERANK_MODELS[rerank_model_id]

    # 生成 rerank model
    export_command_base = "optimum-cli export openvino --model {} --task text-classification".format(
        rerank_model_configuration["model_id"])
    export_command = export_command_base + " " + str(rerank_model_id)

    if not Path(rerank_model_id).exists():
        system(export_command)

