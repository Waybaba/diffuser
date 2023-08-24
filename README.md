# Environment Setup

Before you begin, please ensure that the following environment variables are correctly set: `UDATADIR`, `UPRJDIR`, and `UOUTDIR`. The operations you perform will only modify these three directories on your device.

Here's an example setup:

!!! Create a `.env` file in root dir of this project

```bash
# Put the following in the .env file in the project root
# Example directory paths
export UDATADIR=~/data # directory for dataset
export UPRJDIR=~/code # directory for code
export UOUTDIR=~/output # directory for outputs such as logs
# Example API Key and Worker settings
export WANDB_API_KEY="xxx360492802218be41f76xxxx" # your Weights & Biases API key
export NUM_WORKERS=0 # number of workers to use
# Create directories if they do not exist
mkdir -p $UDATADIR $UPRJDIR $UOUTDIR

```

### Using Docker

If you prefer to use Docker, you can find the `Dockerfile` in the `.devcontainer` directory. Please refer to Docker's documentation if you need guidance on building a Docker image and running a container.

# RUN

### **For `entry - train_diffuser`**

```bash
python entry/entry.py \\
  --experiment=train_diffuser \\
  --trainer.save_freq=100 \\
  --dataset.custom_ds_path="/path/to/data/models/diffuser/d4rl_dataset/maze2d-openlarge-v0-1000000.hdf5"

```

### **For `entry - plan_guided`**

```bash
python entry/entry.py \\
  --experiment=plan_guided \\
  --diffusion.dir="/path/to/outdir/hydra_log/RL_Diffuser/runs/2023-08-23_11-19-12_292485" \\
  --diffusion.epoch=latest \\
  --policy.scale_grad_by_std=true \\
  --guide._target_=diffuser.sampling.NoTrainGuideLonger \\
  --trainer.custom_target=bl2tr \\
  --policy.scale=0.1

```

### **For `entry - train_values`**

```bash
python entry/entry.py \\
  --experiment=train_values \\
  --trainer.save_freq=100 \\
  --dataset.custom_ds_path="/path/to/data/models/diffuser/maze2d-large-1e6FirstGenerate.hdf5"

```