from sentence_transformers import SentenceTransformer
import chromadb

# 1.准备原始文档
documents = [
    'RAG（检索增强生成）是一种先检索再生成的方法。',
    '向量检索会把文本转为向量，然后通过相似度找到相关内容，',
    'Chroma是一个轻量级的向量数据库，适合用来做RAG检索部分。',
    '切分 chunk 会影响检索效果，常见做法是按照段落或者固定长度切分。',
]

# 2.加载embedding模型
model = SentenceTransformer('../models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 3.创建Chroma客户端 内存模式
client = chromadb.Client()

# 4.创建一个collection相当于一个向量表
collection = client.create_collection(name='rag_chromadb_demo')

# 5.把文档存入Chroma
# --Chroma调用embedding函数
# --存储向量
# --绑定原始文本

for i,text in enumerate(documents):
    embedding = model.encode(text).tolist()
    collection.add(
        ids=[str(i)],  # 每条数据的唯一id
        documents=[text],   # 原始文本
        embeddings=[embedding],  # 对应的向量
    )

# 6.用户提问 query
query = '什么是向量检索？'
query_embedding = model.encode(query).tolist()

# 7.使用Chroma进行相似度检索
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

# 8.输出检索结果
print(f'用户输入问题：{query}')
print(f'\n检索到的结果')
for i,doc in enumerate(results['documents'][0]):
    print(f'--Top{i+1}--{doc}')

docs = results['documents'][0]
distances = results['distances'][0]

print(f'检测结果（距离越小越相似）:\n')

for i in range(len(docs)):
    print(f'--Top{i+1}--{docs[i]}')
    print(f'距离：{distances[i]}')
    print('--'*30)















