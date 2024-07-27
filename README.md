# Llama on [Lawrencium] and [Savio]
## Run Llama 3.1 models on Lawrencium
Request a **Jupyter Server on `ES1` A40 GPU node** using [Open OnDemand at https://ood.brc.berkeley.edu/](https://ood.brc.berkeley.edu/).

## Create the Conda Environment
- Please make sure the conda envs directories are on the `$SCRATCH` space, as it needs the scratch space to download Llama 3.1 models 8B (15GiB), 8B-Instruct (15GiB), and 70B (132GiB).
```bash
# Create the conda environment with Python 3.11
conda create --name llama-py311 python=3.11
```

```bash
# Activate the newly created environment
conda activate llama-py311
```

## Install Required Packages
```bash
pip install -r requirements.txt
```

## Add the Conda Environment to Jupyter

```bash
python -m ipykernel install --user --name llama-py311 --display-name "llama-py311"
```

## Access the Llama 3.1 Models via Hugging Face's Transformers

To access the LLAMA 3.1 models, you need to accept the **LLAMA 3.1 COMMUNITY LICENSE AGREEMENT** at [Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B):

1. Click on "Login or Sign Up" to review the conditions.
2. Approval of models access requests usually takes about **2 hours**.

## Set Up Hugging Face on the Cluster

Once you receive the approval email, you can proceed to set up Hugging Face on the cluster:

```bash
huggingface-cli login
```

Follow the instructions to create an access token at [Hugging Face Tokens](https://huggingface.co/settings/tokens) and paste it into the terminal.

## Test the Models
Once you have access to Hugging Face on the GPU node, you can run this Jupyter notebook ([`llama3_1_tests.ipynb`](llama3_1_tests.ipynb)) to test the following models:
- **Llama 3.1 8B** (tested on NVIDIA A40)
- **Llama 3.1 8B-Instruct** (tested on NVIDIA A40)
- **Llama 3.1 70B** (CUDA **out of memory** on NVIDIA A40)

Alternatively, you can test the models by running the following scripts:
- `python single_gpu_inference-llm3_1_8B.py`
- `python single_gpu_inference-llm3_1_8B_Instruct.py`
- `python single_gpu_inference-llm3_1_70B.py`
