from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import API_keys

# 定义模型
model = ChatOpenAI(
    openai_api_key=API_keys['deepseek_api_key'],
    openai_api_base=API_keys['deepseek_url'],
    model=API_keys['deepseek_model'],
    temperature=0.6
)

#  PromptTemplate定义示例格式
example_prompt = PromptTemplate(
    input_variables=['zh', 'en'],  # 示例的变量
    template='中文:{zh} -> 英文:{en}'  # 单个示例的格式
)

translation_examples = [
    {'zh': '吃饭', 'en': 'have a meal'},
    {'zh': '睡觉', 'en': 'sleep'},
    {'zh': '喝水', 'en': 'drink water'}
]
# 初始化FewShortPromptTemplate
few_sort_prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,  # 必传 绑定示例格式
    examples=translation_examples,  # 必传 原始示例数据
    prefix='请参考以下示例完成中英翻译',
    suffix='请翻译{text}',
    input_variables=['text'],
)
# 定义提示词模板
chat_prompt = ChatPromptTemplate.from_messages([
    ('system', '你是一个专业的中英文翻译助手,只输出翻译结果'),
    ('user',few_sort_prompt.format(text='{text}'))
])
# 解析器+ 搭建流水线
parser = StrOutputParser()
chain = chat_prompt | model | parser
# 测试调用
result = chain.invoke({'text':'我很好，谢谢'})
print(result)