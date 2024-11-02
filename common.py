# 参数设置
from pathlib import Path

from llm_config import SUPPORTED_LLM_MODELS

model_language = 'Chinese'
llm_model_id = 'qwen2.5-1.5b-instruct'
enable_awq = 1
embedding_model_id = 'bge-small-zh-v1.5'
rerank_model_id = 'bge-reranker-v2-m3'
embedding_device = 'CPU'
rerank_device = 'CPU'
llm_device = 'CPU'

# compression 设置 这里选了 INT8
model_to_run = 'INT8'
prepare_fp16_model = 0
prepare_int8_model = 1
prepare_int4_model = 0

USING_NPU = embedding_device == "NPU"

llm_model_ids = [model_id for model_id, model_config in SUPPORTED_LLM_MODELS[model_language].items() if model_config.get("rag_prompt_template")]

print(llm_model_ids)

llm_model_configuration = SUPPORTED_LLM_MODELS[model_language][llm_model_id]

print(llm_model_configuration)


pt_model_id = llm_model_configuration["model_id"]
pt_model_name = llm_model_id.split("-")[0]
fp16_model_dir = Path(llm_model_id) / "FP16"
int8_model_dir = Path(llm_model_id) / "INT8_compressed_weights"
int4_model_dir = Path(llm_model_id) / "INT4_compressed_weights"

if model_to_run == "INT4":
    model_dir = int4_model_dir
elif model_to_run == "INT8":
    model_dir = int8_model_dir
else:
    model_dir = fp16_model_dir

