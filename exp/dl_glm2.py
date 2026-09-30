import os, time
os.environ.setdefault("HF_HOME", "/home/ymb/.cache/huggingface")
from huggingface_hub import snapshot_download
t0=time.time()
p = snapshot_download(repo_id="zai-org/glm-4-9b-chat-hf",
                      local_dir="/home/ymb/glm_hf_ok",
                      allow_patterns=["*.safetensors","*.json","*.txt","*.model"],
                      max_workers=4)
print("DOWNLOADED", p, round(time.time()-t0,1), "s")
