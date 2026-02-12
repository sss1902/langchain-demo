# 导包
from config import API_keys
from langchain_core.output_parsers import StrOutputParser  # 解析器
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# 初始化模型
model = ChatOpenAI(
    openai_api_key=API_keys['deepseek_api_key'],
    openai_api_base=API_keys['deepseek_url'],
    model=API_keys['deepseek_model'],
    temperature=0.6
)

# 定义提示词模板
# 提示词模板的好处 灵活 避免硬编码
prompt_template = ChatPromptTemplate.from_messages([
    ('system','你是AI智能助手，用{language}回答问题'),
    ('user','请将以下内容用{language}回答{text}')
])

# 输出解析器
parser = StrOutputParser()
# 构建链 （模板- 模型 解析器）
chain = prompt_template | model | parser

# 测试调用
print(chain.invoke({'language':'法语','text':'解释我爱你的含义'}))
print('*'*20)
print(chain.invoke({'language':'中文','text':'解释我爱你的含义'}))