from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from config import API_keys
from langchain_community.chat_message_histories import ChatMessageHistory

model = ChatOpenAI(
    openai_api_key=API_keys['deepseek_api_key'],
    openai_api_base=API_keys['deepseek_url'],
    model=API_keys['deepseek_model'],
    temperature=0.6
)

# 定义历史存储组件
chat_history = ChatMessageHistory()

# 构建prompt模板
# 基于多角色消息列表快速创建标准化的聊天提示词模板
prompt = ChatPromptTemplate.from_messages([
    ('system','你是友好的助手，用{language}回答问题。对话历史：{history_str}'),
    ('human','{text}')  # 仅保留本轮用户的输入
])

# 构建对话链
conversation_chain = prompt | model | StrOutputParser
# print(conversation_chain)