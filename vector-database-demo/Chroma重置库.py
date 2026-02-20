from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. 初始化中文向量模型
embedding_model = HuggingFaceEmbeddings(model_name="../models/bge-small-zh")

# 2.连接到已经存在的Chroma向量数据库
vectordb = Chroma(persist_directory="./chroma_db",embedding_function=embedding_model)

all_datas = vectordb.get()
print(f"----------------原数据-------------\n{all_datas}")


# 向量库重置
vectordb.reset_collection()
new_all_datas = vectordb.get()
print(f"----------------重置后的数据库-------------\n{new_all_datas}")