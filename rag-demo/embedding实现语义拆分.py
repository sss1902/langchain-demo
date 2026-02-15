from sentence_transformers import SentenceTransformer,util
import re
# A.文档
LONG_DOCUMENT = """
RAG的关键在于检索。切分策略会影响召回质量。
如果切分太粗，检索会不准。如果切分太细，信息又不够。
下面开始讲向量数据库。向量数据库用于存  embedding。
"""

# 第一步 把长文本拆成句子列表
def split_sentences(text):
    # 按照句号，问号，叹号 换行来切
    parts = re.split(r'(?<=[。！？?])|\n+', text)
    # 去掉空的
    return [p.strip() for p in parts if p.strip()]

sentences = split_sentences(LONG_DOCUMENT)
print(sentences)  # ['RAG的关键在于检索。', '切分策略会影响召回质量。', '如果切分太粗，检索会不准。', '如果切分太细，信息又不够。', '下面开始讲向量数据库。', '向量数据库用于存embedding。']
# RAG的关键在于检索，切分策略会影响召回质量，如果切分太粗，检索会不准确，如果切分太细，信息有不够

# 第二步 加载模型 判断
model  = SentenceTransformer('../modelscope-demo/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# 第三步 把每个句子变成向量
embeddings = model.encode(sentences=sentences,convert_to_tensor=True)

# 第四步 开始chunk
threshold = 0.7  # 阈值越大越容易切，chunk块越多

chunks = []
current_chunk = sentences[0]  # 先把第一句放进当前chunk

for i in range(1,len(sentences)):
    # 计算上一句和前一句的相似度（0~1，越大越像）
    # 相似度（理论上-1~1 实际多在0~1，越大越像）
    sim = util.cos_sim(embeddings[i-1],embeddings[i]).item()  # .item()张量的内置方法，将张量转为python原生的数值类型
    print(sim)

    if sim < threshold:
        # 不像：先把当前chunk收起来，然后开新chunk
        chunks.append(current_chunk)
        current_chunk = sentences[i]
    else:
        # 想：把当前句拼到当前chunk后
        current_chunk += sentences[i]

# 循环结束后 把最后一个chunk也收起来
chunks.append(current_chunk)

# 输出
for idx, c in enumerate(chunks,1):
    print(f'\nChunk {idx} | length: {len(c)}')
    print(c)
    print('*'*50)