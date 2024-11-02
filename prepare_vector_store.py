# -*- coding: utf-8 -*-
from llama_index.core import StorageContext, VectorStoreIndex, Settings, load_index_from_storage, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter


from prepare_embedding_model import get_embedding_model

from prepare_llm_model import get_llm_model


def save_to_vector_store(file_path):
    ...

def get_vector_store(index_path):
    # 创建StorageContext对象
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)

    return index

def init_vector_store():
    ...

def parse_text(content):
    with open("doc_emb/text1.txt", "r") as f:
        text = f.readlines()
        print(text)
        return "".join(text)


from common import *

if __name__ == '__main__':
    # 读取文档
    documents = SimpleDirectoryReader("documents").load_data()


    llm = get_llm_model(llm_model_id, model_dir, llm_model_configuration, llm_device)

    Settings.llm = llm

    Settings.embed_model = get_embedding_model(embedding_model_id, False)

    # 对文档进行切分，将切分后的片段转化为embedding向量，构建向量索引
    index = VectorStoreIndex.from_documents(documents,
                                            transformations=[SentenceSplitter(chunk_size=256)])
    # 将embedding向量和向量索引存储到文件中
    index.storage_context.persist(persist_dir='vector-store')


    print("vector save success")

