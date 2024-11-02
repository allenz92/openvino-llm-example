## 测试

```sh
make init # 创建 venv
source bin/activate # 激活 venv
make setup # 下载文件
make deps # 安装依赖
# python  main.py # 执行脚本
````

## 初始化

```sh
make clean
```

## 模型准备
```sh
python prepare_embedding_model.py
python prepare_llm_model.py
python prepare_rerank_model.py
```

## 数据准备
```sh
python prepare_text.py
python prepare_vector_store.py
```

## 推理
```sh
python main.py
```