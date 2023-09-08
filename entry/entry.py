import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True, indicator=["configs"])
import os
import random
import string
import datetime

from time import time
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from typing import Dict, List, Optional, Union, Sequence, Any, Tuple, Callable, TypeVar, Generic, cast, Type, Mapping
from time import sleep
import rich
import rich.syntax
import rich.tree
from pathlib import Path

### check DEBUG mode
# if env variable UDEVICE != docker, set DEBUG=true
print("\n\n### Check Debug")
assert os.environ.get("DEBUG", "noname") != "true", \
	"DEBUG should not be true, or that in all cases data would be loaded from small dataset for debugging."
if os.environ.get("UDEVICEID", "noname") != "docker": # if not docker
	os.environ["DEBUG"] = "true"
	print("Not in docker, setting DEBUG=true ...")
	print("Would skip full data collection and use small dataset for debugging...")
else:
	print("In docker, setting DEBUG=false ...")
	os.environ["DEBUG"] = "false"



### utils

from pytorch_lightning.utilities.logger import _convert_params, _flatten_dict, _sanitize_callable_params

def config_format(cfg: DictConfig) -> DictConfig:
	"""Formats config to be saved to wandb."""
	params = _convert_params(_flatten_dict(_sanitize_callable_params(cfg)))
	return params

def move_all_files(src_dir: str, dst_dir: str) -> None:
	import os
	import shutil

	os.makedirs(dst_dir, exist_ok=True)

	for item in os.listdir(src_dir):
		src_path = os.path.join(src_dir, item)
		dst_path = os.path.join(dst_dir, item)
		shutil.move(src_path, dst_path)

def copy_all_files(src_dir: str, dst_dir: str) -> None:
	import os
	import shutil
	if not os.path.exists(dst_dir):
		os.makedirs(dst_dir)
	
	for item in os.listdir(src_dir):
		src_path = os.path.join(src_dir, item)
		dst_path = os.path.join(dst_dir, item)
		
		if os.path.isdir(src_path):
			shutil.copytree(src_path, dst_path)
		else:
			shutil.copy2(src_path, dst_path)

def print_config_tree(
	cfg: DictConfig,
	print_order: Sequence[str] = (
		"task_name", 
		"tags", 
		"env", 
		"net",
		"policy", 
		"optimizer", 
		"train_collector", 
		"test_collector",
		"trainer",
	),
	resolve: bool = False,
	save_to_file: bool = False,
	) -> None:
	"""Prints content of DictConfig using Rich library and its tree structure.

	Args:
		cfg (DictConfig): Configuration composed by Hydra.
		print_order (Sequence[str], optional): Determines in what order config components are printed.
		resolve (bool, optional): Whether to resolve reference fields of DictConfig.
		save_to_file (bool, optional): Whether to export config to the hydra output folder.
	"""

	style = "dim"
	tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

	queue = []

	# add fields from `print_order` to queue
	for field in print_order:
		queue.append(field) if field in cfg else None

	# add all the other fields to queue (not specified in `print_order`)
	for field in cfg:
		if field not in queue:
			queue.append(field)

	# generate config tree from queue
	for field in queue:
		branch = tree.add(field, style=style, guide_style=style)
		config_group = cfg[field]
		if isinstance(config_group, DictConfig):
			branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
		else:
			branch_content = str(config_group)

		branch.add(rich.syntax.Syntax(branch_content, "yaml"))

	# print config tree
	rich.print(tree)

	# save config tree to file
	if save_to_file:
		with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
			rich.print(tree, file=file)

def move_output_to_wandb_dir(src_dir, dest_dir):
	print("\n\n###")
	print("Moving output to wandb dir ...")
	print(f"From: {src_dir}")
	print(f"To: {dest_dir}")
	copy_all_files(src_dir, dest_dir)
	print("Moving wandb done!")

def pre_start_check(cfg):
	# print environment variables
	print("\n\n\n### Printing environment variables ...")
	# HYDRA_FULL_ERROR
	print("HYDRA_FULL_ERROR: ", os.environ["HYDRA_FULL_ERROR"])
	# print where torch gpu is available
	import torch
	print("torch.cuda.is_available(): ", torch.cuda.is_available())
	# print username
	# print("username: ", os.environ["USER"])
	import subprocess
	result = subprocess.run(["whoami"], capture_output=True, text=True)
	print("username: ", result.stdout.strip())

### main functions

def initialize_wandb(cfg):
	# Generate a unique directory name with a timestamp and 10 random characters
	timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	random_chars = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
	unique_dir = f"{timestamp}-{random_chars}"

	# Create unique wandb directory
	if cfg.wandb.buf_dir:
		# amlt_output_dir = os.environ['AMLT_OUTPUT_DIR'] if "AMLT_OUTPUT_DIR" in os.environ else None
		amlt_output_dir = os.environ['AMLT_DIRSYNC_DIR'] if "AMLT_DIRSYNC_DIR" in os.environ else None
		wandb_dir_prefix = amlt_output_dir if amlt_output_dir else os.path.join(root, "output")
		wandb_dir = os.path.join(wandb_dir_prefix, unique_dir)  
		print("NOTE: Using wandb buffer dir: ", wandb_dir)
	else: 
		wandb_dir = cfg.output_dir

	os.makedirs(wandb_dir, exist_ok=True)

	wandb.init(
		project=cfg.task_name,
		tags=cfg.tags,
		config=config_format(cfg),
		dir=wandb_dir,
		mode=cfg.wandb.mode,
		id = cfg.output_dir.replace("/","_")
	)
	return wandb_dir

def close_wandb(wandb_dir, cfg):
	# wandb.alert(title="Run Finish!", text=f"cfg.tags: {cfg.tags}", level=wandb.AlertLevel.INFO)
	wandb.finish()

	# Move output to wandb dir if necessary
	if cfg.wandb.buf_dir:
		retry = 10
		time_to_sleep = 5
		for i in range(retry):
			try:
				move_output_to_wandb_dir(wandb_dir,cfg.output_dir)
				break
			except Exception as e:
				print(f"Failed to move output to wandb dir. Retrying ({i+1}/{retry}) ...")
				print(e)
				sleep(time_to_sleep)

def seed_everything(seed):
	import random, os
	import numpy as np
	import torch
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	# torch.backends.cudnn.deterministic = True # this would slow down training
	# torch.backends.cudnn.benchmark = False # this would slow down training

def save_config(cfg, path):
	import os
	import yaml
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w") as file:
		yaml.dump(OmegaConf.to_container(cfg, resolve=True), file, default_flow_style=False)

def link_output(cfg):
	"""
		link the output_dir to ./debug/latest/output/ for easier access
		if there exists a previous link, delete it first
		note that the parent dir could miss, so create it if necessary
	"""
	import os
	from pathlib import Path

	output_dir = Path(cfg.paths.output_dir)
	root = Path.cwd()
	latest_dir = root / "debug" / "latest" / "output"
	
	latest_dir.parent.mkdir(parents=True, exist_ok=True)

	if latest_dir.is_symlink():
		latest_dir.unlink()
	
	os.symlink(output_dir, latest_dir)

	print(f"Linked output_dir to {latest_dir}")

def pre_debug(cfg):
	print("###"*9999)
	print("\n\n\n### Listing all files in cfg.data_dir ...")
	import glob
	for file in glob.glob(cfg.paths.data_dir+"/*"):
		print(file)

@hydra.main(version_base=None, config_path=str(root / "configs"), config_name="entry.yaml")	
def main(cfg):
	if not os.path.exists(root / ".env"):
		raise FileNotFoundError("Please create .env file in the root directory. See .env.example for reference.")

	# pre print
	pre_start_check(cfg)

	# link output dir for easier access
	link_output(cfg)

	print("\n\n\n### Printing Hydra config ...")
	print_config_tree(cfg, resolve=True)
	save_config(cfg, cfg.output_dir+"/hydra_config.yaml")

	if 0:
		pre_debug(cfg)


	print("\n\n\n### Initializing wandb ...")
	try:
		wandb_dir = initialize_wandb(cfg)
	except Exception as e:
		print("Exception caught! when initializing wandb ...")
		print("This is a fatal error, main code would only run if wandb is initialized successfully.")
		raise e

	print("\n\n\n### Initializing and running Hydra config ...")
	cfg = hydra.utils.instantiate(cfg)

	print("\n\n\n### Seed everything ...")
	seed = int(time()) if cfg.seed is None else cfg.seed
	seed_everything(seed)

	print("\n\n\n### Initializing and running runner ...")
	cfg.runner().start(cfg)

	print("\n\n\n### Closing wandb ...")
	close_wandb(wandb_dir, cfg)

	print("Done!")

if __name__ == "__main__":
	main()
