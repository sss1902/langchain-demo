# 第一步 导入核心模块
import chromadb
# 统计每个文档选了几条时，不用先检查文档是否在字典里，直接加1就行，默认从0就开始计算
from collections import defaultdict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

import config
from config import API_keys

# 第二步 加载多文档+贴唯一ID，防乱第一步
# 模拟真实业务文档
DOCUMENTS = [
    {
        "doc_id": "doc_001",  # 唯一ID（自定义，不重复即可）
        "title": "RAG 基础规则",
        "text": "RAG 的核心是「先检索，再回答」，能大幅减少AI幻觉。检索的文本越相关，回答越准确。"
    },
    {
        "doc_id": "doc_002",
        "title": "RAG 工程规范",
        "text": "多文档 RAG 必须保留文档元数据，拆分文本时要继承原文档的ID。单文档检索结果最多保留2条，避免霸屏。"
    },
    {
        "doc_id": "doc_003",
        "title": "RAG 避坑指南",
        "text": "如果不给AI标注内容来源，AI会混淆不同文档的信息；如果不存元数据，检索后无法溯源，必出问题。"
    }
]

# 打印验证，确认文档加载并带id
print('----------步骤2 加载文档完成--------------')
for doc in DOCUMENTS:
    print(f'文档id:{doc["doc_id"]} | 标题：:{doc["title"]}')

# 第三步 拆分文档+保留id 防乱第二步
# 逻辑
# --用最新的拆分器把长文档拆成小文本块（适合检索），且每个小块都记成原文档的id和标题

# 初始化最新半段拆分器
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,  # 块与块之间不重叠
    separators=['\n', '，', '。']
)

# 存储拆分后的小块 带元数据
split_chunks = []
for doc in DOCUMENTS:
    # 拆分文档为小块
    text_parts = splitter.split_text(doc["text"])
    # 给每个小块继承源文档id和标题
    for idx, part in enumerate(text_parts):
        split_chunks.append({
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "chunk_id": f'{doc["doc_id"]}_{idx}',
            'content': part.strip()
        })

# 打印验证
print(f'\n----------步骤3 拆分文档完成------------------')
for chunk in split_chunks:
    print(f'小块ID:{chunk["chunk_id"]} | 来自:{chunk["title"]} | 内容“:{chunk["content"]}')



# 第四步 存入向量数据库+绑定元数据
# 初始化向量库
chroma_client = chromadb.Client()
# 创建多文档RAG专用集合
collection = chroma_client.create_collection(name='multi-doc_rag')

# 提取存入数据 内容+ID+元数据
chunk_contents=[c['content'] for c in split_chunks]
chunk_ids = [c['chunk_id'] for c in split_chunks]
chunk_meta_datas = [
    {'doc_id':c['doc_id'],'title':c['title']}
    for c in split_chunks
]

# 存入向量数据库
collection.add(
    documents=chunk_contents,
    ids=chunk_ids,
    metadatas=chunk_meta_datas
)

# 打印验证
print('\n---------步骤4 存入向量数据库完成----------------')
print(f'向量数据库总小块数：{collection.count()}')

# C:\Users\86158\.cache\chroma\onnx_models\all-MiniLM-L6-v2\onnx.tar.gz:   6%|▌
# | 4.37M/79.3M [00:13<03:10, 412kiB/s]




# 第五步 检索+防霸屏（防乱第四步）
# 先召回足够多的相关小块
user_query = '多文档RAG怎么设计才不会乱？'
# 5.1向量检索，召回10条相关小块
retrieval_results = collection.query(
    query_texts=[user_query],  # 用户问题
    n_results=10,  # 多召回，后续过滤
    include=['documents','metadatas']  # 必须返回内容+元数据
)

# 5.2 过滤规则
#  -- 规则 单个文档最多两条
max_per_doc = 2
filtered_results = []
doc_count = defaultdict(int)  # 统计每个文档的命中数

# 遍历结果，按照规则过滤
for content, meta in zip(retrieval_results['documents'][0], retrieval_results['metadatas'][0]):
    if doc_count[meta['doc_id']] < max_per_doc:
        filtered_results.append({
            'content': content,
            'title': meta['title'],
        })
        doc_count[meta['doc_id']] += 1


# 打印验证
print('\n---------步骤5 检索+过滤完成----------------')
print(f'原始检索结果数：{len(retrieval_results["documents"][0])}')
print(f'过滤后结果数：{len(filtered_results)}')
print(f'各文档取数：{dict(doc_count)}')  # 验证每个文档最多2条











































