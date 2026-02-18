# 导入库
import json
from langchain_chroma import Chroma  # 最新Chroma向量数据库
from langchain_huggingface import HuggingFaceEmbeddings  # 中文文本嵌入
from sqlalchemy.testing.suite.test_reflection import metadata

# 1.初始化中文嵌入模型
embedding_model = HuggingFaceEmbeddings(model_name="../models/bge-small-zh")

# 2.定义一个向量数据库的存储路径
persist_dir = "./chroma_db"

# 3.读取题库
with open(file="questions.json", mode="r", encoding="utf-8") as f:
    data = json.load(f)
    print(data)

texts,metadatas,ids = [],[],[]

for item in data:
    text = (
        f"题目: {item['question']}\n"
        f"选项: {' '.join(item['options'])}\n"
        f"答案: {item['answer']}\n"
        f"解析: {item['explanation']}\n"
    )
    print(text)
    texts.append(text)
    metadatas.append({"id": item["id"],"question":item["question"],"answer":item["answer"]})
    ids.append(item["id"])

# 4.初始化Chroma并添加文本
# Chroma 内存模式和持久化模式
vectordb = Chroma(persist_directory=persist_dir,embedding_function=embedding_model)
vectordb.add_texts(texts=texts,metadatas=metadatas,ids=ids)
print("向量库初始化并存入数据完成")
