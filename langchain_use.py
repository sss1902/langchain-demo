import os
import langchain
from langchain_openai import ChatOpenAI  # 需要tiktoken
from langchain_core.messages import HumanMessage, SystemMessage
from config import API_keys

# 查看langchain的版本
print(f'langchain的版本为{langchain.__version__}')  # langchain的版本为1.1.3

# 方式一 等待完全输出后再获取数据
# 方式一适用场景适用模型一次性输出完的
from langchain_core.output_parsers import StrOutputParser

# 创建模型实例
print('*********方式一**************')
model1 = ChatOpenAI(
    openai_api_key=API_keys['deepseek_api_key'],
    openai_api_base=API_keys['deepseek_url'],
    model=API_keys['deepseek_model'],
    temperature=0.6
)

# 发送消息 等待模型完全生成输出
response1 = model1.invoke([
    SystemMessage(content='请将内容翻译成英文，无需给任何翻译之外的内容'),
    HumanMessage(content='你好')
])


# 打印模型完整输出
print('模型完整输出： ')
print(response1.content)

print('*********方式二**************')
# 方式二 流式输出也就是生成器的方式
stream_response = model1.stream([
    SystemMessage(content='请将内容翻译成英文，无需给任何翻译之外的内容'),
    HumanMessage(content='你好大米你好你好你好你好你好你好你好你好你好你好你好你好你好你好你好')
])
print('流式输出')
full_text = ''
for chunk in stream_response:
    text = chunk.content
    print(text, end='', flush=True)  # 实时打印 flush=True立即刷新缓冲区，把当前内容输出到终端
    full_text += text
print('\n流式输出完成，最终结果为：')
print(full_text)