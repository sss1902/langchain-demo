import json

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. 初始化中文向量模型
embedding_model = HuggingFaceEmbeddings(model_name="../models/bge-small-zh")

# 2.连接到已经存在的Chroma向量数据库
vectordb = Chroma(persist_directory="./chroma_db",embedding_function=embedding_model)

print(vectordb.get())


# with open(file="questions.json", mode="r", encoding="utf-8") as f:
#     data = json.load(f)
#
# # 拼接成
# texts = []
# ids = []
# metadatas=[]
#
# for item in data:
#     text = (f"题目: {item['question']}\n"
#             f"选项: {' '.join(item['options'])}"
#             f"答案: {item['answer']}"
#             f"解析: {item['explanation']}")
#     texts.append(text)
#
#     #
#     ids.append(item["id"])
#
#     # 元数据





# 添加数据库
# vectordb.add_texts(texts=texts,ids=ids)

# 查询数据

# 更新
updated_doc = Document(
    # 新的文档内容（替换原有内容）
    page_content="""题目: 以下哪种水果富含维生素C？
选项: A.香蕉 B.苹果 C.橙子 D.西瓜
答案: C
解析: 橙子属于柑橘类水果，每100g橙子含维生素C约53mg，远高于香蕉、苹果等水果。""",
    # 可选：元数据（可新增/修改字段，若不需要可留空 {}）
    metadata={
        "type": "选择题",
        "subject": "生活常识",
        "difficulty": "简单",
        "id": "q001"  # 元数据中保留ID，方便管理
    }
)

vectordb.update_document(document_id='q001',document=updated_doc)

print(vectordb.get())