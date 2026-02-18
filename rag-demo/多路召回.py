# BM25关键词召回+向量召回
# 导入相关库
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# 模拟知识库
DOCUMENTS = [
    {"id": "doc1", "text": "Albert Einstein was a German-born theoretical physicist."},
    {"id": "doc2", "text": "He developed the theory of relativity."},
    {"id": "doc3", "text": "Einstein is widely considered the father of modern physics."},
    {"id": "doc4", "text": "Marie Curie was a Polish and naturalised-French physicist."},
    {"id": "doc5", "text": "She conducted pioneering research on radioactivity."},
    {"id": "doc6", "text": "Curie was the first woman to win a Nobel Prize."},
]



# 实现BM25召回器
class SimpleBM25Retriever:
    def __init__(self, documents):
        self.documents = documents
        # 对文本进行分词
        self.corpus = [doc["text"] for doc in documents]
        # 使用rank-bm5库训练BM25模型
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query, top_k=2):
        # 1.对查询进行分词
        tokenized_query = query.split(" ")
        # 计算每个文档的相关性得分
        scores = self.bm25.get_scores(tokenized_query)
        # 根据得分排序，取Top k
        top_indices = np.argsort(scores)[::-1][:top_k]
        # 返回文档和得分
        results = [(self.documents[i], scores[i]) for i in top_indices if scores[i] > 0]
        return results
        