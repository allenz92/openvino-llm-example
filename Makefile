SHELL := $(shell which bash)
# 导入 api key, cookie 等
# 模版参考 local.tpl.d
-include local/env.mk

src := \
https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py \
https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/pip_helper.py \
https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/llm_config.py

text_example_en.pdf := https://github.com/openvinotoolkit/openvino_notebooks/files/15039728/Platform.Brief_Intel.vPro.with.Intel.Core.Ultra_Final.pdf
text_example_cn.pdf := https://github.com/openvinotoolkit/openvino_notebooks/files/15039713/Platform.Brief_Intel.vPro.with.Intel.Core.Ultra_Final_CH.pdf
files := text_example_en.pdf text_example_cn.pdf

requires :=     llama-index \
    faiss-cpu \
    pymupdf \
    langchain \
    llama-index-readers-file \
    llama-index-vector-stores-faiss \
    llama-index-llms-langchain \
    llama-index-llms-huggingface>=0.3.0,<0.3.4 \
    llama-index-embeddings-huggingface>=0.3.0 \
	git+https://github.com/huggingface/optimum-intel.git \
	git+https://github.com/openvinotoolkit/nncf.git \
	datasets accelerate gradio ipython

require_pres := openvino>=2024.2 openvino-tokenizers[transformers]>=2024.2
require_nodeps := llama-index-llms-openvino>=0.3.1 llama-index-embeddings-openvino>=0.2.1 llama-index-postprocessor-openvino-rerank>=0.2.0

all: setup deps prepare run

run:
	python main.py

setup:
	for x in $(src); do  \
		[[ -f `basename $${x}` ]] || wget "$${x}"; \
	done
	$(foreach x,$(files),[[ -f $(x) ]] || wget $($(x)) -O $(x);)
	mkdir -p documents

devices:
	python -c 'import openvino as ov;core = ov.Core(); print(core.available_devices)'


clean:
	rm -rf bin include lib lib64 __pycache__ etc share documents

init:
	python -m venv .

deps:
	pip install $(foreach x,$(requires),'$(x)')
	$(foreach x,$(require_pres),pip install --pre -U '$(x)';)
	$(foreach x,$(require_nodeps),pip install --no-deps '$(x)';)

prepare/%:
	python prepare_$*.py

# 需要准备的
prepares := embedding_model llm_model rerank_model \
			text convert vector_store

prepare: $(foreach x,$(prepares),prepare/$(x)) 

test:
	:

tar_excludes = $(wildcard $(shell cat .gitignore)) .git src.tgz
	
tar:
	tar $(foreach x,$(tar_excludes),--exclude $(x)) -czf src.tgz .
