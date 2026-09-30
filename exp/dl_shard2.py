import os, time, shutil, hashlib
os.environ.setdefault("HF_HOME", "/home/ymb/.cache/huggingface")
from huggingface_hub import hf_hub_download
t0=time.time()
p = hf_hub_download(repo_id="zai-org/glm-4-9b-chat-hf",
                    filename="model-00002-of-00004.safetensors",
                    local_dir="/home/ymb/glm_local/hf_shard2",
                    revision="8599336fc6c125203efb2360bfaf4c80eef1d1bf")
print("DOWNLOADED", p, round(time.time()-t0,1), "s", flush=True)
h=hashlib.sha256()
with open(p,'rb') as f:
    for b in iter(lambda: f.read(1<<22), b''): h.update(b)
print("SHA256", h.hexdigest(), flush=True)
print("EXPECT 029329f34bb73bd71b7e5f6391c62383d6bf888b88d4168fb7c416ea283da3f7", flush=True)
print("MATCH", h.hexdigest()=="029329f34bb73bd71b7e5f6391c62383d6bf888b88d4168fb7c416ea283da3f7", flush=True)
