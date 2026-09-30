# 发布检查环境

这是已有 `qwen35` 环境的已安装版本记录，不是经过全新环境安装验证的依赖锁。模型、CUDA 驱动和数据需另行准备。

```text
torch==2.10.0+cu128
transformers==4.57.6
vllm==0.17.0
numpy==2.2.6
pandas==2.3.3
datasets==4.8.3
pyarrow==23.0.1
sacrebleu==2.6.0
rouge-score==0.1.2
nltk==3.10.3
scikit-learn==1.7.2
scipy==1.15.3
sentence-transformers==5.4.1
safetensors==0.7.0
huggingface-hub==0.36.2
requests==2.32.5
```

Laya 源码版本：`4066d5d5fbf08b66c6757ddeedbd797bd7655bc0`，包版本 `0.3.20`；使用仓库自带的 `exp/vendor/laya`。项目的 acceptance 适配代码会将这个目录加入导入路径。

本次将 `pytest==9.1.1` 和 `ruff==0.16.9` 安装到独立临时目录用于发布检查，没有改动上述实验环境。绘图脚本还需要 matplotlib，COMET 复评依赖独立的 COMET 环境，这两项未包含在本表中。
