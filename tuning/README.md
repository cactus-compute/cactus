# Tuning

Fine-tuning Gemma models for Cactus using Google Tunix.

## Notebooks

- `lora_gemma.ipynb` - LoRA fine-tuning
- `grpo_gemma.ipynb` - GRPO (Group Relative Policy Optimization)
- `dpo_gemma.ipynb` - DPO (Direct Preference Optimization)

## GCE VM Setup for Fine-Tuning

### 1. Create TPU VM

Create a v5litepod-8 TPU VM in GCE:
- SW version: `v2-alpha-tpuv5-lite`
- Name: `v5-8`

Reference: [TPU Runtime Versions](https://docs.cloud.google.com/tpu/docs/runtimes?hl=en&_gl=1*1tpeg3j*_ga*MTk1NzE5MjMyNy4xNzYwOTEwNjk3*_ga_WH2QY8WWF5*czE3NjIxNTU1OTEkbzE3JGcwJHQxNzYyMTU1NTkxJGo2MCRsMCRoMA..#training-v5p-v5e)

### 2. Configure VM

SSH into the VM using the supplied gcloud command, then run:

```bash
# Create .env file with required credentials
vim .env

# Download and install Anaconda
curl -O https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
bash ~/Anaconda3-2025.06-0-Linux-x86_64.sh  # always input "yes"/enter
source ~/.bashrc

# Create conda environment (Python 3.12 - MUST BE 12, NOT 11!)
conda create -n colab python=3.12 -y
conda activate colab

# Install dependencies
pip install 'ipykernel<7' jupyterlab
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install --upgrade clu
```

Reference: [Run JAX on TPU](https://docs.cloud.google.com/tpu/docs/run-calculation-jax)

Exit the SSH session after setup is complete.

### 3. Connect from Local Machine

From your local machine, run the following to connect to Jupyter Lab:

```bash
gcloud compute tpus tpu-vm ssh v5-8 --zone=us-west1-c \
  -- -L 8080:localhost:8080 -L 6006:localhost:6006 \
  "source \$HOME/anaconda3/etc/profile.d/conda.sh && \
   conda activate colab && \
   jupyter lab \
     --ServerApp.allow_origin='https://colab.research.google.com' \
     --port=8080 \
     --no-browser \
     --ServerApp.port_retries=0 \
     --ServerApp.allow_credentials=True"
```

Reference: [Local Runtimes in Colab](https://research.google.com/colaboratory/local-runtimes.html)

### 4. Environment Variables

Example `.env` file:

```bash
HF_TOKEN=
KAGGLE_USERNAME=
KAGGLE_KEY=
WANDB_API_KEY=
```

## Notes

- **IMPORTANT**: Use `%pip` not `!pip` in notebooks!
- Python 3.12 is the recommended version
