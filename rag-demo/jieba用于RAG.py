import jieba

# 定义领域词
# 这些词在RAG、BM25中很重要，不希望被拆分
DOMAIN_WORDS = [
    'RAG系统',
    '向量数据库',
    '大语言模型',
    '知识检索增强',
    'Embedding模型',
    '混合检索'
]

# 把领域词加入到jieba
for word in DOMAIN_WORDS:
    jieba.add_word(word)


# 分词函数 给BM25用
def tokenize(text: str) -> list[str]:
    """
    规则：
    -使用精确模式jieba.lcut
    -只做轻度清洗（去掉标点和空白）
    :param text:
    :return:
    """

    # 分词
    tokens = jieba.lcut(text)

    # 只去掉明显没有意义的符号
    drop = {
        ' ', '\n', '\t', ',', ',', '，', '。'
    }

    tokens = [t for t in tokens if t not in drop and t.strip()]
    return tokens


# 演示文本是怎么被拆分的
if __name__ == '__main__':
    # 示例1 普通文本拆分
    text = 'RAG系统通常结合向量数据库和大语言模型，实现知识检索增强。'
    print('原文',text)
    print('分词结果',tokenize(text=text))
