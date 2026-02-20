# 导入相关库
# faiss不是数据库
import faiss
import numpy as np

# 准备数据
# 10000个句子的向量，每个向量长度是768维---这个是bert模型的常见维度
database_vectors = np.random.rand(10_000,768).astype("float32")
print(database_vectors)

print("\n")


# 创建索引
# 基于内积的索引  内积越大，向量的相似度越高
index = faiss.IndexFlatIP(768)  # 768是维度
print(index)

# 归一化
faiss.normalize_L2(database_vectors)

# 把向量加进索引
index.add(database_vectors)
print(f"已加载{index.ntotal}个向量")  # number total

# 准备我们要搜索的新向量
query_vector = np.random.rand(1,768).astype("float32")

# 归一化
faiss.normalize_L2(query_vector)

# 执行搜索
similarities,indices = index.search(query_vector, k=3)

print("--------------结果-----------------")
print(similarities)
print(indices)