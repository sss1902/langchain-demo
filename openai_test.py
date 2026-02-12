from config import API_keys

# 导入OpenAI客户端库
from openai import OpenAI

client = OpenAI(api_key=API_keys['deepseek_api_key'], base_url=API_keys['deepseek_url'])

# 调用 OpenAI API生成聊天对话机器人对话的回复
response = client.chat.completions.create(
    model=API_keys['deepseek_model'],
    messages=[
        {
            'role': 'system',
            'content': 'You are a helpful assistant'
        },
        {
            'role': 'user',
            'content': '你好'
        }
    ],
    stream=False,
    n=3  # 核心参数 指定三个候选回复  但是deepseek只支持1
)
print(response)
# print(response.choices[0])
# print(response.choices[0].message.content)

# print('***************************************************')
# for i, choice in enumerate(response):
#     print(f'第{i}个候选回复: {choice.message.content}')
#     print('----------分割线----------')
