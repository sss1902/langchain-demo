#
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# 1.用 PromptTemplate定义示例格式
example_prompt = PromptTemplate(
    input_variables=['zh', 'en'],  # 示例的变量
    template='中文:{zh} -> 英文:{en}'  # 单个示例的格式
)
# 2.准备原始示例数据
translation_examples = [
    {'zh': '吃饭', 'en': 'have a meal'},
    {'zh': '睡觉', 'en': 'sleep'},
    {'zh': '喝水', 'en': 'drink water'}
]
# 3.初始化FewShortPromptTemplate
few_sort_prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,  # 必传 绑定示例格式
    examples=translation_examples,  # 必传 原始示例数据
    prefix='请参考以下示例完成中英翻译',
    suffix='请翻译{text}',
    input_variables=['text'],
)

# 4.格式化生成完整的提示词
final_prompt = few_sort_prompt.format(text='你是一个乐于助人的助手')
print(f'拼接后的完整的提示词 \n {final_prompt}')