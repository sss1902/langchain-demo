from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. 初始化中文向量模型
embedding_model = HuggingFaceEmbeddings(model_name="../models/bge-small-zh")

# 2.连接到已经存在的Chroma向量数据库
vectordb = Chroma(persist_directory="./chroma_db",embedding_function=embedding_model)


# 查看所有数据
all_datas = vectordb.get()
print(f"----------------原数据-------------\n{all_datas}")


# 新数据
new_item = {
    "id": "q003",
    "question": "太阳系中最大的行星是？",
    "options": ["A. 地球", "B. 木星", "C. 金星", "D. 火星"],
    "answer": "B",
    "explanation": "木星体积最大。"
}

text = (
    f"题目: {new_item['question']}\n"
    f"选项: {' '.join(new_item['options'])}\n"
    f"答案: {new_item['answer']}\n"
    f"解析: {new_item['explanation']}"
)

metadata = {"id": new_item["id"], "question": new_item["question"], "answer": new_item["answer"]}
vectordb.add_texts(texts=[text], metadata=[metadata],ids=[new_item["id"]])
new_all_datas = vectordb.get()
print("添加完成")
print(f"----------------添加后的数据-------------\n{new_all_datas}")

