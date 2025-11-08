## MetaGpro

This is the official implementation of "MPAS: Breaking Sequential Constraints of Multi-Agent Communication Topologies via Individual-Epistemic Message Propagation"

Our multi-agent scheme is named as MPAS (Message Passing Multi-Agent System).

We will further fix the bugs, update the applications of MPAS in the future.

We refer to the <ins>GPTSwarm</ins> library to build our agentic graphs **src/** [https://github.com/metauto-ai/GPTSwarm](https://github.com/metauto-ai/GPTSwarm).

We refer to the <ins>G-Designer</ins> library to build our basic agents **src/** [https://github.com/yanweiyue/GDesigner](https://github.com/yanweiyue/GDesigner).


<p align="center">
<img center src="https://github.com/rkxuan/MPAS/scheme.png" width = "450" alt="logo">
</p>

## Basic Environment
* `CUDA == 12.1`
* `Python == 3.10` 
* `PyTorch == 2.1.2+cu121`

## You should add your API keys in .env
```python
OPENAI_API_KEY=""  # for OpenAI LLM beckend
BASE_URL="" # for URL
```

## Dataset

We use seven widely used benchmark datasets: MULL, GSM8K, SVAMP, MultiArith, AQuA, HumanEval, MBPP

## Code Structure

dataset: The benchmark datasets

experiment: The codes for pre-processing datasets and running benchmark evaluations

mpma/environment: The specific definations of agents

mpma/llm: The APIs to call LLMs

mpma/optimizer: The implementation of parameterized probabilistic distribution over node-wise topologies

mpma/system: The implementation of the node-wise multi-agent system

mpma/utils: Other needed functions or classes

## Implementation  

MMLU: 
* `!python experiment/run_mmlu.py --debug=True`

AQuA: 
* `!python experiment/run_aqua.py --debug=True`

## News
* 🔥 **[2025/11/08]** Our work has been accepted by [AAAI2026](Oral).
