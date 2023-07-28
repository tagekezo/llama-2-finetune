# Sample code to finetune llama-2

## Requirements

It takes 3 hours to run this on AWS using g5.12xlarge instance (4x Nvidia A10 GPUs).

```python
huggingface-cli login # requires authentication to download llama-2 models
git clone git@github.com:tagekezo/llama-2-finetune.git
cd llama-2-finetune
source activate pytorch
python -m venv venv # create virtual env
pip install -r req.txt
nohup python -u ./main.py > main.log 2>&1 & # run as background process
tail -f main.log # monitor run process
```

 This solution references [falcon-40b-qlora-finetune-summarize](https://github.com/aws-samples/amazon-sagemaker-generativeai/blob/main/studio-notebook-fine-tuning/falcon-40b-qlora-finetune-summarize.ipynb). The main changes are the model id pointing to llama-2 and target_modules for loraconfig.  

