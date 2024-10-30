from pathlib import Path
from notebook_utils import optimize_bge_embedding
import os
import shutil
system = os.system

from llm_config import (
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_RERANK_MODELS,
    SUPPORTED_LLM_MODELS,
)

# 参数设置
model_language = 'English'
llm_model_id = 'qwen2.5-7b-instruct'
enable_awq = 1
embedding_model_id = 'bge-large-en-v1.5'
rerank_model_id = 'bge-reranker-large'
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


print(embedding_model_id)
embedding_model_configuration = SUPPORTED_EMBEDDING_MODELS[model_language][embedding_model_id]

print(embedding_model_configuration)

# 生成 embedding_model
export_command_base = "optimum-cli export openvino --model {} --task feature-extraction".format(embedding_model_configuration["model_id"])
export_command = export_command_base + " " + str(embedding_model_id)

if not Path(embedding_model_id).exists():
    system(export_command)

rerank_model_configuration = SUPPORTED_RERANK_MODELS[rerank_model_id]

# 生成 rerank model
export_command_base = "optimum-cli export openvino --model {} --task text-classification".format(rerank_model_configuration["model_id"])
export_command = export_command_base + " " + str(rerank_model_id)

if not Path(rerank_model_id).exists():
    system(export_command)

# 如果启用 NPN 
npu_embedding_dir = embedding_model_id + "-npu"
npu_embedding_path = Path(npu_embedding_dir) / "openvino_model.xml"

if USING_NPU and not Path(npu_embedding_dir).exists():
    shutil.copytree(embedding_model_id, npu_embedding_dir)
    optimize_bge_embedding(Path(embedding_model_id) / "openvino_model.xml", npu_embedding_path)


from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding

embedding_model_name = npu_embedding_dir if USING_NPU else embedding_model_id
batch_size = 1 if USING_NPU else 4

embedding = OpenVINOEmbedding(
    model_id_or_path=embedding_model_name, embed_batch_size=batch_size, device=embedding_device, model_kwargs={"compile": False}
)
if USING_NPU:
    embedding._model.reshape(1, 512)
embedding._model.compile()

embeddings = embedding.get_text_embedding("Hello World!")


from llama_index.postprocessor.openvino_rerank import OpenVINORerank

reranker = OpenVINORerank(model_id_or_path=rerank_model_id, device=rerank_device, top_n=2)


from llama_index.llms.openvino import OpenVINOLLM

import openvino.properties as props
import openvino.properties.hint as hints
import openvino.properties.streams as streams


if model_to_run == "INT4":
    model_dir = int4_model_dir
elif model_to_run == "INT8":
    model_dir = int8_model_dir
else:
    model_dir = fp16_model_dir
print(f"Loading model from {model_dir}")

ov_config = {hints.performance_mode(): hints.PerformanceMode.LATENCY, streams.num(): "1", props.cache_dir(): ""}

stop_tokens = llm_model_configuration.get("stop_tokens")
completion_to_prompt = llm_model_configuration.get("completion_to_prompt")

if "GPU" in llm_device and "qwen2-7b-instruct" in llm_model_id:
    ov_config["GPU_ENABLE_SDPA_OPTIMIZATION"] = "NO"

# On a GPU device a model is executed in FP16 precision. For red-pajama-3b-chat model there known accuracy
# issues caused by this, which we avoid by setting precision hint to "f32".
if llm_model_id == "red-pajama-3b-chat" and "GPU" in core.available_devices and llm_device in ["GPU", "AUTO"]:
    ov_config["INFERENCE_PRECISION_HINT"] = "f32"

llm = OpenVINOLLM(
    model_id_or_path=str(model_dir),
    context_window=3900,
    max_new_tokens=2,
    model_kwargs={"ov_config": ov_config, "trust_remote_code": True},
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    completion_to_prompt=completion_to_prompt,
    device_map=llm_device,
)

response = llm.complete("introduce intel core")
print(str(response))



from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.faiss import FaissVectorStore
from transformers import StoppingCriteria, StoppingCriteriaList
import faiss
import torch

if model_language == "English":
    text_example_path = "text_example_en.pdf"
else:
    text_example_path = "text_example_cn.pdf"


class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids):
        self.token_ids = token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


if stop_tokens is not None:
    if isinstance(stop_tokens[0], str):
        stop_tokens = llm._tokenizer.convert_tokens_to_ids(stop_tokens)
    stop_tokens = [StopOnTokens(stop_tokens)]

loader = PyMuPDFReader()
documents = loader.load(file_path=text_example_path)

# dimensions of embedding model
d = embedding._model.request.outputs[0].get_partial_shape()[2].get_length()
faiss_index = faiss.IndexFlatL2(d)
Settings.embed_model = embedding

llm.max_new_tokens = 2048
if stop_tokens is not None:
    llm._stopping_criteria = StoppingCriteriaList(stop_tokens)
Settings.llm = llm

vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    transformations=[SentenceSplitter(chunk_size=200, chunk_overlap=40)],
)

query_engine = index.as_query_engine(streaming=True, similarity_top_k=10, node_postprocessors=[reranker])
if model_language == "English":
    query = "What can Intel vPro® Enterprise systems offer?"
else:
    query = "英特尔博锐® Enterprise系统提供哪些功能？"

streaming_response = query_engine.query(query)
streaming_response.print_response_stream()
