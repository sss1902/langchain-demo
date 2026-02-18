from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. 初始化中文向量模型
embedding_model = HuggingFaceEmbeddings(model_name="../models/bge-small-zh")

# 2.连接到已经存在的Chroma向量数据库
vectordb = Chroma(persist_directory="./chroma_db",embedding_function=embedding_model)

# 3.用户查询
query  = "地球上哪个海洋是最大的？"
# 4.
results = vectordb.similarity_search(query=query,k=1)

results_with_score = vectordb.similarity_search_with_score(query=query,k=2)  # 这个是距离值 距离越小越相似

# 打印查询结果
for i in results:
    print("内容：\n",i.page_content)
    print("元数据：",i.metadata)

    print(i)

print("-----------------------")
print(results_with_score)