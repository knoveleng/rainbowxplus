# RainbowXPlus

This repository contains the codebase for the paper RainbowXPlus [Comming soon]

## Setup

To set up the environment and run the program, follow the steps below:

### 1. Create a Virtual Environment

Create a Python virtual environment and install the required dependencies.

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

### 2. Configure Hugging Face Token (Optional)

If certain resources from the Hugging Face Hub are required, configure your API token.

```bash
export HF_AUTH_TOKEN="YOUR_HF_TOKEN"
```

or
```bash
huggingface-cli login --token=YOUR_HF_TOKEN
```

### 3. Run the Program

After setting up the environment and configurations, run the program by specifying config file. For example

```bash
python -m rainbowxplus.rainbowxplus --config_file configs/base.yml --dataset data/harmfulQA.json --model Qwen/Qwen2.5-7B-Instruct
```

## Evaluation
After running the program, the results will be saved in the `./logs` directory. To evaluate, run:

```bash
python -m rainbowxplus.evaluate --config configs/eval.yml -l ./logs/model_name/dataset_name -a rainbowxplus -m method
```

where `method` is one of:
- `all` for all scores (Standard RainbowXPlus)
- `median` for median score (RainbowXPlus - $\alpha$)
- `max` for max score (RainbowXPlus - $\beta$)


## Cite our Work
Comming soon ...