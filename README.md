
# Environment Setup
Before you begin, please ensure that the following environment variables are correctly set: UDATADIR, UPRJDIR, and UOUTDIR. The operations you perform will only modify these three directories on your device.

Here's an example setup:
```
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
If you prefer to use Docker, you can find the Dockerfile in the .devcontainer directory. Please refer to Docker's documentation if you need guidance on building a Docker image and running a container.