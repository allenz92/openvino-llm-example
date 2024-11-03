# README.MD

## 快速开始

1. 按照 local.tpl.d 配置 密钥 cookie 等

```sh
make init # 初始化 venv
source bin/activate # 激活 venv
make
```

## 具体细节

### 环境准备

```sh
make init # 创建 venv
source bin/activate # 激活 venv
make setup # 下载文件
make deps # 安装依赖
````

```

### 模型准备
```sh
python prepare_embedding_model.py
python prepare_llm_model.py
python prepare_rerank_model.py
```

### 数据准备
```sh
python prepare_convert.py # int8
python prepare_text.py
python prepare_vector_store.py
```

### 推理
```sh
python main.py
```

### 清除环境

```sh
make clean
```
