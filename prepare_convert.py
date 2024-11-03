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


