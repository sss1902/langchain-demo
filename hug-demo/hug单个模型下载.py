# 下载BAAI/bge-small-zh模型
import os

local_model_path = "../models/bge-small-zh"  # 本地存放模型的目录
# 如果模型不存在，则下载
if not os.path.exists(local_model_path):
    from transformers import AutoModel,AutoTokenizer
    print("模型不存在，开始下载到本地...")
    AutoTokenizer.from_pretrained("BAAI/BGE-small-zh").save_pretrained(local_model_path)
    AutoModel.from_pretrained("BAAI/BGE-small-zh").save_pretrained(local_model_path)
    print("模型下载完成")