import os
from pathlib import Path
import requests
import shutil
import io

# fetch model configuration

config_shared_path = Path("../../utils/llm_config.py")
config_dst_path = Path("llm_config.py")
text_example_en_path = Path("text_example_en.pdf")
text_example_cn_path = Path("text_example_cn.pdf")
text_example_en = "https://github.com/openvinotoolkit/openvino_notebooks/files/15039728/Platform.Brief_Intel.vPro.with.Intel.Core.Ultra_Final.pdf"
text_example_cn = "https://github.com/openvinotoolkit/openvino_notebooks/files/15039713/Platform.Brief_Intel.vPro.with.Intel.Core.Ultra_Final_CH.pdf"

if not config_dst_path.exists():
    if config_shared_path.exists():
        try:
            os.symlink(config_shared_path, config_dst_path)
        except Exception:
            shutil.copy(config_shared_path, config_dst_path)
    else:
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/llm_config.py")
        with open("llm_config.py", "w", encoding="utf-8") as f:
            f.write(r.text)
elif not os.path.islink(config_dst_path):
    print("LLM config will be updated")
    if config_shared_path.exists():
        shutil.copy(config_shared_path, config_dst_path)
    else:
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/llm_config.py")
        with open("llm_config.py", "w", encoding="utf-8") as f:
            f.write(r.text)


if not text_example_en_path.exists():
    r = requests.get(url=text_example_en)
    content = io.BytesIO(r.content)
    with open("text_example_en.pdf", "wb") as f:
        f.write(content.read())

if not text_example_cn_path.exists():
    r = requests.get(url=text_example_cn)
    content = io.BytesIO(r.content)
    with open("text_example_cn.pdf", "wb") as f:
        f.write(content.read())
