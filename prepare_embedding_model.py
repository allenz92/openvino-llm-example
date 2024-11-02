import shutil
import os
from pathlib import Path

system = os.system
from llm_config import (
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_RERANK_MODELS,
    SUPPORTED_LLM_MODELS,
)

from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding

from notebook_utils import optimize_bge_embedding

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


def get_embedding_model(embedding_model_id, using_npu):
    # 如果启用 NPN
    npu_embedding_dir = embedding_model_id + "-npu"
    npu_embedding_path = Path(npu_embedding_dir) / "openvino_model.xml"

    if using_npu and not Path(npu_embedding_dir).exists():
        shutil.copytree(embedding_model_id, npu_embedding_dir)
        optimize_bge_embedding(Path(embedding_model_id) / "openvino_model.xml", npu_embedding_path)

    embedding_model_name = npu_embedding_dir if using_npu else embedding_model_id
    batch_size = 1 if using_npu else 4

    embedding_device = "NPU" if using_npu else "CPU"

    embedding = OpenVINOEmbedding(
        model_id_or_path=embedding_model_name, embed_batch_size=batch_size, device=embedding_device,
        model_kwargs={"compile": False}
    )
    if using_npu:
        embedding._model.reshape(1, 512)
    embedding._model.compile()

    return embedding


if __name__ == '__main__':
    print(embedding_model_id)
    embedding_model_configuration = SUPPORTED_EMBEDDING_MODELS[model_language][embedding_model_id]

    print(embedding_model_configuration)

    # 生成 embedding_model
    export_command_base = "optimum-cli export openvino --model {} --task feature-extraction".format(
        embedding_model_configuration["model_id"])
    export_command = export_command_base + " " + str(embedding_model_id)

    if not Path(embedding_model_id).exists():
        system(export_command)

