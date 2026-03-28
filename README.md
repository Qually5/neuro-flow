# neuro-flow

**A high-performance framework for distributed LLM fine-tuning and inference.**

## Features
- Distributed training with PyTorch FSDP
- Low-latency inference using vLLM integration
- Automated hyperparameter optimization
- Support for multi-modal model architectures

## Installation
```bash
pip install neuro-flow
```

## Usage
```python
from neuro_flow import Trainer

trainer = Trainer(model='llama-3-8b', dataset='custom_data')
trainer.train()
```