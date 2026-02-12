from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

from config import API_keys


# 先自己定义一个函数
# 1.定义天气查询的函数（兼容city和__arg1参数）
def get_weather(city: str = None, __arg1: str = None):
    target_city = city or __arg1
    return {'杭州': '15℃ 多云', '上海': '18℃ 晴'}.get(target_city, '未查到城市')


# 2.封装langchain天气工具
weather_tool = Tool(
    name='get_weather',
    func=get_weather,
    description='查询城市天气，入参为城市名（如杭州、上海）'
)

# 3.初始化模型
model = ChatOpenAI(
    openai_api_key=API_keys['deepseek_api_key'],
    openai_api_base=API_keys['deepseek_url'],
    model=API_keys['deepseek_model'],
    temperature=0.6
)
# 4.模型绑定工具
model_with_tool = model.bind_tools([weather_tool])

# 5.构建用户问题
user_msg = HumanMessage(content='杭州今天的天气怎么样？')

# 6.模型生成工具调用指令
# 让具备“识别工具”能力的模型，看完用户的问题后，告诉你“该用哪个工具，传什么参数”，并把这些信息打包给你
# call_result只有工具调用指令，没有工具执行结果
call_result = model_with_tool.invoke([user_msg])
# print(call_result)
# content='我来帮您查询杭州今天的天气情况。'
# additional_kwargs={'tool_calls': [{'id': 'call_00_U76Dx3jJae9OU7HQqvPGgKrp',
# 'function': {'arguments': '{"__arg1": "杭州"}', 'name': 'get_weather'},
# 'type': 'function', 'index': 0}],
# 'refusal': None} response_metadata={'token_usage': {'completion_tokens': 54, 'prompt_tokens': 323, 'total_tokens': 377, 'completion_tokens_details': None,
# 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 323}, 'model_name': 'deepseek-chat',
# 'system_fingerprint': 'fp_eaab8d114b_prod0820_fp8_kvcache', 'id': '167a9072-d1a4-4625-a407-94566d860d69', 'service_tier': None, 'finish_reason': 'tool_calls', '
# logprobs': None} id='lc_run--019c5167-1912-7d93-998f-c78f2fbb60d4-0' tool_calls=[{'name': 'get_weather', 'args': {'__arg1': '杭州'},
# 'id': 'call_00_U76Dx3jJae9OU7HQqvPGgKrp', 'type': 'tool_call'}] invalid_tool_calls=[]
# usage_metadata={'input_tokens': 323, 'output_tokens': 54, 'total_tokens': 377, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}


# 从模型给我们的打包信息里，挑出要使用的工具叫什么名字
tool_name = call_result.tool_calls[0]['name']
# 从模型给我们的打包信息里，挑出调用这个工具需要给什么参数
tool_args = call_result.tool_calls[0]['args']
# print(tool_args)
# 从模型给我们的打包信息里，挑出这次工具调用的专属编号
tool_call_id = call_result.tool_calls[0]['id']

# 7.执行工具获取结果
# 用提取到的工具参数，手动调用天气工具 获取工具执行后的结果
# tool_result = model_with_tool.invoke([tool_name, tool_call_id])
tool_result = weather_tool.func(** tool_args)
# print(tool_result)

# 8.封装工具结果为标准格式
# 把工具执行结果和对应的专属编号打包成标准格式，方便模型识别这是哪次工具调用返回的结果
tool_msg = ToolMessage(content=tool_result,tool_call_id=tool_call_id)
print(tool_msg)
print('\n')
# 模型生成最终的回答
final_answer = model.invoke([user_msg,call_result,tool_msg])
print(final_answer)
# 打印
print('\n')
print(f'最终回答{final_answer.content}')