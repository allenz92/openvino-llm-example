from pathlib import Path

from llama_index.llms.openvino import OpenVINOLLM

import openvino.properties as props
import openvino.properties.hint as hints
import openvino.properties.streams as streams

def get_llm_model(llm_model_id, model_dir, llm_model_configuration, llm_device):
    print(f"Loading model from {model_dir}")

    ov_config = {hints.performance_mode(): hints.PerformanceMode.LATENCY, streams.num(): "1", props.cache_dir(): ""}

    completion_to_prompt = llm_model_configuration.get("completion_to_prompt")

    if "GPU" in llm_device and "qwen2-7b-instruct" in llm_model_id:
        ov_config["GPU_ENABLE_SDPA_OPTIMIZATION"] = "NO"


    llm = OpenVINOLLM(
        model_id_or_path=str(model_dir),
        context_window=3900,
        max_new_tokens=2,
        model_kwargs={"ov_config": ov_config, "trust_remote_code": True},
        generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
        completion_to_prompt=completion_to_prompt,
        device_map=llm_device,
    )

    return llm

# response = llm.complete("introduce intel core")
# print(str(response))