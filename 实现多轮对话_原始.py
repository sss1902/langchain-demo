from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from config import API_keys

# 初始化模型
# 定义模型
model = ChatOpenAI(
    openai_api_key=API_keys['deepseek_api_key'],
    openai_api_base=API_keys['deepseek_url'],
    model=API_keys['deepseek_model'],
    temperature=0.6
)

parser = StrOutputParser()

#
# 聊天记录列表
history = []

# 多轮对话函数
def record_and_invoke(user_input, language='English'):
    global history  # global只在函数内部使用，使得函数内部可以修改全局变量
    
