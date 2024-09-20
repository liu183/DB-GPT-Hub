import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir='/group_share/models', revision='master')



# curl -X POST "http://127.0.0.1:6007" \
#      -H 'Content-Type: application/json' \
#      -d '{"prompt": "你好"}'