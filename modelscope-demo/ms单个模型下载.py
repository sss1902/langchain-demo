from modelscope import snapshot_download
import os

# 定义要保存的模型路径
model_dir = './models'

# 如果目录不存在。则创建
os.makedirs(model_dir, exist_ok=True)

# 下载模型 只下载文件 不运行
# msd的model_id必须是命名空间 / 名称 的格式
model_path = snapshot_download(model_id='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',cache_dir=model_dir)

print(f'model downloaded to {model_path}')