import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True, indicator=["configs"])
from collections import namedtuple
import numpy as np
from tqdm import tqdm
from diffuser.datasets.preprocessing import get_preprocess_fn
from diffuser.datasets.normalization import get_normalizer
from diffuser.datasets.d4rl import load_environment
from pytorch_lightning import LightningDataModule
from typing import Any, Dict, Optional, Tuple
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
import torch
import os
import random
from copy import deepcopy
import hashlib

Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')
TransitionBatch = namedtuple('TransitionBatch', 's s_ act')
EpisodeBatch = namedtuple('EpisodeBatch', 'trajectories conditions')
EpisodeValidBatch = namedtuple('EpisodeValidBatch', 'trajectories conditions valids')
MUJOCO_ENVS = ["hopper", "walker2d", "halfcheetah"]

### functions

class DatasetNormalizerW:
	def __init__(self, dataset, normalizer):
		normalizer = get_normalizer(normalizer)
		# self.norm_keys = list(dataset.keys())
		self.normalizers = {}
		for k, v in dataset.items():
			self.normalizers[k] = normalizer(v)
		
	def normalize(self, x, key):
		return self.normalizers[key].normalize(x)

	def unnormalize(self, x, key):
		return self.normalizers[key].unnormalize(x)

def load_kuka(env, custom_ds_path=None):
	""" load kuka env 
	"""
	from glob import glob
	assert "kuka" in env, "only support kuka env"
	if custom_ds_path is None:
		custom_ds_path = "/data/models/diffuser/d4rl_dataset/kuka/kuka_dataset/"
		print(f"using kuka default dataset path {custom_ds_path}")
	from gym_stacking.env import StackEnv
	env = StackEnv()
	dataset = custom_ds_path + "/*.npy"
	# dataset = "/data/models/diffuser/d4rl_dataset/kuka/kuka_dataset/*.npy" # DEBUG
	datasets = sorted(glob(dataset))
	print(f"found {len(datasets)} datasets at {dataset}")
	datasets = [np.load(dataset) for dataset in tqdm(
		datasets[:100] if os.environ.get("DEBUG", "false").lower()=="true" else datasets,
	)] # read from file
	if os.environ.get("DEBUG", "false").lower()=="true":
		print("\n### debug mode is on, only load 100 datasets !!!\n")
	datasets = [dataset[::2] for dataset in datasets]
	ep_lengths = [len(dataset) for dataset in datasets]
	qstates = np.concatenate(datasets, axis=0)

	# qstates = np.zeros((max_n_episodes, max_path_length, obs_dim))
	# path_lengths = np.zeros(max_n_episodes, dtype=np.int)

	# for i, dataset in enumerate(datasets):
	# 	qstate = np.load(dataset)
	# 	qstate = qstate[::2]
	# 	print(qstate.max(), qstate.min())
	# 	# qstate[np.isnan(qstate)] = 0.0
	# 	path_length = len(qstate)

	# 	if path_length > max_path_length:
	# 		qstates[i, :max_path_length] = qstate[:max_path_length]
	# 		path_length = max_path_length
	# 	else:
	# 		qstates[i, :path_length] = qstate
	# 	path_lengths[i] = path_length
	# qstates = qstates[:i+1]
	# path_lengths = path_lengths[:i+1]
	# return qstates, path_lengths
	terminals = np.zeros_like(qstates[:,0])
	terminals[np.cumsum(ep_lengths)-1] = 1
	dataset = {
		"observations": qstates,
		"actions": np.random.randn(*qstates.shape)[:,:11], # act_dim = 11
		"terminals": terminals
	}
	return env, dataset


### dataset

class EnvDataset:
	""" A common dr4l dataset API
		! TODO use padding
		Common process:
			normalization
			into GPU
		# Args: 
			mode:
				transition: load (s,a,s) transition pairs as
					TransitionBatch{
						s: (N, obs_dim)
						s_: (N, obs_dim)
						act: (N, act_dim)
					}
				episode: load Episode as
					EpisodeBatch{
						trajectories: (N, horizon, obs_dim + act_dim)
						conditions: (N, horizon, obs_dim)
					}
			env: 
			custom_ds_path: ""
			preprocess_fns: []
			normalizer: 'LimitsNormalizer' or 'StandardNormalizer'
			gpu: True
			seed: None # TOODO

			only_start_end_episode: false for HER
			horizon: length of the trajectory
				LimitsNormalizer: normalize to [-1, 1]
	"""
	def __init__(self, 
			  env, 
			  custom_ds_path=None,
			  preprocess_fns=[],
			  normalizer='LimitsNormalizer', 
			  gpu=True,
			  seed=None, 
			  *args,
			  **kwargs,
		):
		assert type(env) == str, "env should be a string"
		assert "maze" in env or "halfcheetah" in env or "kuka" in env or "walker2d" in env or "hopper" in env, "maze envs not supported, since d4rl does not provide terminal"

		### get dataset (setup self.dataset, self.env)
		self.env_name = env
		if "kuka" in self.env_name:
			self.env, self.dataset = load_kuka(self.env_name, custom_ds_path)
		elif self.env_name.endswith("mixed"):
			# e.g. halfcheetah-mixed
			base_name = self.env_name.split("-")[0]
			ds_suffix_list = ["random", "medium", "expert"]
			ds_list = []
			for suffix in ds_suffix_list:
				env_name_ = base_name + "-" + suffix + "-v0"
				env_ = load_environment(env_name_)
				ds = env_.get_dataset()
				ds_list.append(ds)
			self.env, self.dataset = env_, {}
			for k in ds_list[0].keys():
				self.dataset[k] = np.concatenate([ds[k] for ds in ds_list], axis=0)
		else:
			self.env = load_environment(self.env_name) # TOODO can not use gym.make ?
			if custom_ds_path: self.dataset = self.env.get_dataset(custom_ds_path)
			else: self.dataset = self.env.get_dataset()
			# self.dataset = d4rl.qlearning_dataset(self.env_name)
			# self.dataset.update(self.env.get_dataset())

		### pre_process fns
		assert preprocess_fns == "by_env", "only support by_env"
		if "maze" in self.env_name: preprocess_fns = ["maze2d_set_terminals"]
		elif [self.env_name.startswith(v) for v in ["halfcheetah", "walker2d", "hopper"]].count(True) == 1: 
			preprocess_fns = []
		elif "kuka" in self.env_name: preprocess_fns = []
		else: raise NotImplementedError("env not supported")
		self.preprocess_fn = get_preprocess_fn(preprocess_fns, self.env_name) # TOODO do not use original function
		self.dataset = self.preprocess_fn(self.dataset)
		
		### remove keys
		KEYS_NEED = ["observations", "actions", "terminals", "timeouts"]
		KEYS_NEED += ["infos/qpos", "infos/qvel", "infos/action_lop_probs"] # TOODO for controller reset position
		keys_to_delete = [k for k in self.dataset.keys() if k not in KEYS_NEED]
		for k in keys_to_delete: del self.dataset[k]
		
		### normalize
		if normalizer == "by_env":
			if "maze" in self.env_name: normalizer = "LimitsNormalizer"
			elif [self.env_name.startswith(v) for v in ["halfcheetah", "walker2d", "hopper"]].count(True) == 1: 
				normalizer = "GaussianNormalizer" # DebugNormalizer, GaussianNormalizer
			elif "kuka" in self.env_name: normalizer = "LimitsNormalizer"
			else: raise NotImplementedError(f"env {self.env_name} not supported")
		else:
			normalizer = normalizer
		self.observation_dim = self.dataset['observations'].shape[1]
		self.action_dim = self.dataset['actions'].shape[1]
		self.normalizer = DatasetNormalizerW(self.dataset, normalizer)
		for k in ["observations", "actions"]:
			self.dataset[k] = self.normalizer.normalize(self.dataset[k], k)
		
		### put into GPU
		if gpu:
			for k in self.dataset.keys():
				# if double, turn to float
				if self.dataset[k].dtype == np.float64:
					self.dataset[k] = self.dataset[k].astype(np.float32)
				self.dataset[k] = torch.tensor(self.dataset[k]).cuda()
		
		### set renderer
		if "maze" in self.env_name:
			from diffuser.utils.rendering import Maze2dRenderer
			self.renderer = Maze2dRenderer(self.env_name)
		elif [self.env_name.startswith(v) for v in ["halfcheetah", "walker2d", "hopper"]].count(True) == 1:
			from diffuser.utils.rendering import MuJoCoRenderer
			self.renderer = MuJoCoRenderer(self.env_name)
		elif "kuka" in self.env_name:
			from denoising_diffusion_pytorch.utils.rendering import KukaRenderer
			self.renderer = KukaRenderer()
		else:
			raise NotImplementedError("env not supported")

		### lazyload indices
		self.indices = self.lazyload_indices()

	def __len__(self):
		return len(self.indices)

	def get_episodes_ref(self, num_episodes=4):
		"""
		Retrieves reference episodes from the dataset.

		Each episode is represented as a dictionary with the following structure:
		{
			"s": (T, obs_dim),      # states
			"act": (T, act_dim),    # actions
			...
		}

		Returns the first {num_episodes} episodes from the dataset. The returned values are unnormalized.

		Returns:
			list: A list of dictionaries, each representing a reference episode.
		"""
		dataset = self.dataset
		
		# Find indices where the episodes terminate
		dones = dataset["terminals"]
		if "timeouts" in dataset: dones |= dataset["timeouts"]
		termination_indices = torch.where(dones)[0]
		
		# Randomly select {num_episodes} starting indices for the episodes
		random_start_indices = np.random.randint(0, len(termination_indices) - 3, num_episodes)
		episode_boundaries = torch.stack([termination_indices[random_start_indices], 
										termination_indices[random_start_indices + 1]], dim=1)
		
		# Construct the list of reference episodes
		reference_episodes = []
		for start_idx, end_idx in episode_boundaries:
			episode_data = {
				"s": dataset["observations"][start_idx:end_idx-1],
				"act": dataset["actions"][start_idx:end_idx-1],
				"s_": dataset["observations"][start_idx + 1:end_idx -1 + 1]
			}
			
			# Include additional keys starting with "infos/"
			for key, value in dataset.items():
				if key.startswith("infos/"):
					episode_data[key[6:]] = value[start_idx:end_idx]
			
			reference_episodes.append(episode_data)

		# Convert to CPU and numpy format, if needed
		for episode in reference_episodes:
			for key, value in episode.items():
				if torch.is_tensor(value):
					episode[key] = value.cpu().numpy()

		# Unnormalize the data
		for episode in reference_episodes:
			episode["s"] = self.normalizer.unnormalize(episode["s"], "observations")
			episode["s_"] = self.normalizer.unnormalize(episode["s_"], "observations")
			episode["act"] = self.normalizer.unnormalize(episode["act"], "actions")

		# Ensure all episodes have the same length
		min_length = min(len(episode["s"]) for episode in reference_episodes)
		for episode in reference_episodes:
			for key, value in episode.items():
				if isinstance(value, np.ndarray):
					episode[key] = value[:min_length]
		
		self.episodes_ref = reference_episodes
		return self.episodes_ref

	def lazyload_indices(self):
		""" lazyload indices
		save indices with pickle after first-time processing
		hash self.kwargs as the unique name for the indices
		"""
		assert os.environ.get("UDATADIR") is not None, "UDATADIR not set"
		save_dir = os.path.join(os.environ.get("UDATADIR"), "models", "diffuser", "d4rl_dataset", "indices_buf")
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		cfgs = deepcopy(self.kwargs)
		cfgs["debug"] = os.environ.get("DEBUG", "false").lower() == "true"
		if cfgs["debug"]: print("\n### DEBUG is on !!! Only load part data!")
		if "forcesave" in cfgs: del cfgs["forcesave"]
		if "lazyload" in cfgs: del cfgs["lazyload"]
		hash_name = hashlib.md5(str(cfgs).encode()).hexdigest()
		file_path = os.path.join(save_dir, hash_name + ".pkl")

		if "lazyload" in self.kwargs and self.kwargs["lazyload"] and os.path.exists(file_path):
			print(f"\n[EnvDataset] loading indices from {file_path}\n")
			indices = torch.load(file_path)
		else:
			if "lazyload" not in self.kwargs:
				msg = "lazyload not found, would load ..."
			else:
				msg = "will always remake indices" if not self.kwargs["lazyload"] else f"can not find {file_path}, will remake indices"
			print(msg)
			indices = self.make_indices()
			if not os.path.exists(file_path) or ("forcesave" in self.kwargs and self.kwargs["forcesave"]):
				torch.save(indices, file_path)
				print(f"\n[EnvDataset] indices saved to {file_path}\n")

		print(f"indices length is {len(indices)}\n")
		return indices

class EnvEpisodeDataset(EnvDataset):

	def __init__(self, *args, **kwargs):
		self.kwargs = kwargs
		super().__init__(*args, **kwargs)

	def make_indices(self):
		"""
			makes indices for sampling from dataset;
			each index maps to a datapoint
			(N, 2)
			each element is (start, end)
		"""
		dataset = self.dataset
		# fast_idx_making = True
		if self.kwargs["mode"] == "default":
			dones = dataset["terminals"]
			if "timeouts" in dataset: dones |= dataset["timeouts"]
			dones_idxes = torch.where(dones)[0]
			indices = []
			ep_start = 0
			print("making indexes ...")
			for i in tqdm(range(len(dones_idxes))):
				if dones_idxes[i] > ep_start: 
					if dones_idxes[i] - ep_start >= self.kwargs["horizon"]:
						indices.append([ep_start, dones_idxes[i]])
						if os.environ.get("DEBUG", "false").lower()=="true" and len(indices) > 10000: return torch.tensor(indices)
					ep_start = dones_idxes[i] + 1
			indices = torch.tensor(indices)
		elif self.kwargs["mode"].startswith("multi_step"):
			""" make indices with different values of interval
			for each episode, 
			[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
			samples the ones 
			indices: [(start, interval)]
				e.g. [(0, 2)] means sequence [0, 2, 4, 6, ..., 2*(horizon-1)]
			ps. need to make sure intervals are balanced
			"""
			indices = []
			mode, multi_step = self.kwargs["mode"].split("%")
			multi_step = int(multi_step)
			dones = dataset["terminals"]
			if "timeouts" in dataset: dones |= dataset["timeouts"]
			dones_idxes = torch.where(dones)[0]
			ep_start = 0
			max_gap = multi_step * self.kwargs["horizon"]
			for ep_start in tqdm(range(0, len(dones) - max_gap)):
				for inter in range(1, int(multi_step) + 1):
					indices.append([ep_start, ep_start + self.kwargs["horizon"] * inter, inter])
					if os.environ.get("DEBUG", "false").lower()=="true" and len(indices) > 10000: return torch.tensor(indices)
			indices = torch.tensor(indices)
		elif self.kwargs["mode"].startswith("ep_multi_step"):
			"""
				same as before but is episode based
				would not use indices cross episodes
			"""
			indices = []
			mode, multi_step = self.kwargs["mode"].split("%")
			multi_step = int(multi_step)
			dones = dataset["terminals"]
			if "timeouts" in dataset: dones |= dataset["timeouts"]
			dones_idxes = torch.where(dones)[0]
			
			print("making indexes for episode-based multi-step ...")
			ep_start = 0
			max_gap = multi_step * self.kwargs["horizon"]
			for ep_end in tqdm(dones_idxes):
				for i in range(ep_start, ep_end - max_gap + 1):
					for inter in range(1, multi_step + 1):
						indices.append([i, i + self.kwargs["horizon"] * inter, inter])
				ep_start = ep_end + 1  # Move to the start of the next episode
				if os.environ.get("DEBUG", "false").lower()=="true" and len(indices) > 10000: return torch.tensor(indices)
				
			indices = torch.tensor(indices)
		elif self.kwargs["mode"].startswith("valid_multi_step"):
			"""
				same as before but is episode based
				would use indices cross episodes but mark the ones in the 
				later as invalid
				[(start, end, interval, invalid_start)]
			"""
			indices = []
			mode, multi_step = self.kwargs["mode"].split("%")
			multi_step = int(multi_step)
			dones = dataset["terminals"]
			if "timeouts" in dataset: dones |= dataset["timeouts"]
			dones_idxes = torch.where(dones)[0]
			
			print("making indexes for valid episode-based multi-step ...")
			ep_start = 0
			max_gap = multi_step * self.kwargs["horizon"]
			for ep_end in tqdm(dones_idxes):
				for i in range(ep_start, ep_end):
					for inter in range(1, multi_step + 1):
						item_end = i + self.kwargs["horizon"] * inter
						invalid_start = max((ep_end-i)//inter, self.kwargs["horizon"]-1)
						indices.append([i, item_end, inter, invalid_start])
				ep_start = ep_end + 1  # Move to the start of the next episode
				if os.environ.get("DEBUG", "false").lower()=="true" and len(indices) > 10000: return torch.tensor(indices)
			
			indices = torch.tensor(indices)
		else:
			raise NotImplementedError("mode not supported")

		return indices

	def get_conditions(self, observations):
		'''
			condition on both the current observation and the last observation in the plan
		'''
		cond = {0: observations[0]}
		# if "maze" in self.env_name:
		# 	cond.update({
		# 		self.kwargs["horizon"] - 1: observations[-1],
		# 	})
		return cond
	
	def __getitem__(self, idx):
		""" 
		"""
		if self.kwargs["mode"] == "default":
			### B random slice
			start, end = self.indices[idx]
			start = np.random.randint(start, end - self.kwargs["horizon"] + 1)
			end = start + self.kwargs["horizon"]
			observations = self.dataset["observations"][start:end]
			actions = self.dataset["actions"][start:end]

			###  ! DEBUG random flip observation to make two way
			# if np.random.rand() > 0.5:
			# 	observations = self.flip_trajectory(observations)
			# 	actions = self.flip_trajectory(actions)

			conditions = self.get_conditions(observations)
			trajectories = torch.cat([actions, observations], axis=-1)
			batch = EpisodeBatch(trajectories, conditions)
		elif self.kwargs["mode"].startswith("multi_step") or self.kwargs["mode"].startswith("ep_multi_step"):
			start, end, inter = self.indices[idx]
			observations = self.dataset["observations"][start:end:inter]
			actions = self.dataset["actions"][start:end:inter]
			conditions = self.get_conditions(observations)
			trajectories = torch.cat([actions, observations], axis=-1)
			batch = EpisodeBatch(trajectories, conditions)
		elif self.kwargs["mode"].startswith("valid_multi_step"):
			start, end, inter, invalid_start = self.indices[idx]
			observations = self.dataset["observations"][start:end:inter]
			actions = self.dataset["actions"][start:end:inter]
			valids = torch.ones_like(observations[:,0])
			valids[invalid_start:] = 0
			conditions = self.get_conditions(observations)
			trajectories = torch.cat([actions, observations], axis=-1)
			batch = EpisodeValidBatch(trajectories, conditions, valids)

		return batch

	def flip_trajectory(self, observations):
		""" make another way to observations
		observatios: T,2
		"""
		start = observations[0]
		end = observations[-1]
		mid = (start + end) / 2
		# find the mirror point by mid
		observations = 2 * mid - observations
		# reverse
		observations = observations.flip(0)
		return observations

class EnvTransitionDataset(EnvDataset):

	def __init__(self, *args, **kwargs):
		self.kwargs = kwargs
		super().__init__(*args, **kwargs)

	def make_indices(self):
		"""
		Generate indices for sampling.
		
		Parameters:
		- dataset: dict
		  The dataset containing 'observations' and 'terminals'.
		- multi_step: int
		  The step size for sampling.

		Returns:
		- np.array
		  A NumPy array containing valid index pairs.
		  indices: (N*multi_step, 2)
		  	[start, end) does not include any dones==True
		"""
		dataset, multi_step = self.dataset, self.kwargs["multi_step"]

		num_data = len(dataset["observations"])
		dones = dataset.get("terminals", np.zeros(num_data, dtype=bool))
		if "timeouts" in dataset: dones |= dataset["timeouts"]

		valid_indices = torch.where(dones == 0)[0]
		
		indices = []
		print("making indices ...")
		for i in tqdm(valid_indices.cpu().numpy()):
			for j in range(1, multi_step + 1):
				end_idx = i + j
				if end_idx >= num_data: break
				if dones[end_idx]:
					indices.append([i, end_idx])
					break
				indices.append([i, end_idx])
				if os.environ.get("DEBUG", "false").lower()=="true" and len(indices) > 10000: return np.array(indices)


		return np.array(indices)
	
	def __getitem__(self, idx):
		start, end = self.indices[idx]
		s = self.dataset["observations"][start]
		s_ = self.dataset["observations"][end]
		act = self.dataset["actions"][start]
		return TransitionBatch(s, s_, act)

### datamodule

class EnvDatamodule(LightningDataModule):
	def __init__(self, **kwargs) -> None:
		"""Initialize a `MNISTDataModule`.

		:param data_dir: The data directory. Defaults to `"data/"`.
		:param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
		:param batch_size: The batch size. Defaults to `64`.
		:param num_workers: The number of workers. Defaults to `0`.
		:param pin_memory: Whether to pin memory. Defaults to `False`.
		"""
		super().__init__()

		# this line allows to access init params with 'self.hparams' attribute
		# also ensures init params will be stored in ckpt
		self.save_hyperparameters(logger=False)

		# data transformations
		self.transforms = transforms.Compose(
			[transforms.ToTensor()]
		) # TODO controller

		self.data_train: Optional[Dataset] = None
		self.data_val: Optional[Dataset] = None
		self.data_test: Optional[Dataset] = None

		self.setup()
		self.setup_info()

	def prepare_data(self) -> None:
		"""Download data if needed. Lightning ensures that `self.prepare_data()` is called only
		within a single process on CPU, so you can safely add your downloading logic within. In
		case of multi-node training, the execution of this hook depends upon
		`self.prepare_data_per_node()`.

		Do not use it to assign state (self.x = y).
		"""
		# predownload
		print("prepare_dataset ...")
		self.hparams.dataset()

	def	setup_info(self):
		self.info = {
			"obs_dim": self.dataset.env.observation_space.shape[0],
			"act_dim": self.dataset.env.action_space.shape[0],
			"env": self.dataset.env,
			"dataset": self.dataset,
			"data_train": self.data_train,
			"data_val": self.data_val,
			"data_test": self.data_test,
			"renderer": self.dataset.renderer
		}
	
	def setup(self, stage: Optional[str] = None) -> None:
		"""Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

		This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
		`trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
		`self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
		`self.setup()` once the data is prepared and available for use.

		:param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
		"""
		# load and split datasets only if not loaded already
		if not self.data_train and not self.data_val and not self.data_test:
			assert type(list(self.hparams.train_val_test_split)) == list, "train_val_test_split should be a list"
			self.dataset = self.hparams.dataset()
			train_val_test_split = self.hparams.train_val_test_split \
				if type(self.hparams.train_val_test_split[0]) != float else \
				[max(int(len(self.dataset) * split),self.hparams.batch_size) for split in self.hparams.train_val_test_split]
			train_val_test_split[0] = len(self.dataset) - sum(train_val_test_split[1:])
			self.data_train, self.data_val, self.data_test = random_split(
				dataset=self.dataset,
				lengths=train_val_test_split,
				# generator=torch.Generator().manual_seed(0),
			)
			self.data_val = [self.data_val]
			self.data_test = [self.data_test]
			# print length information
			print("[Dataset length][dslen,dataset length, val_len]: train {}, val {}, test {}".format(
				len(self.data_train), len(self.data_val[0]), len(self.data_test[0])
			))
			
	def train_dataloader(self):
		# return empty dataloader
		# get empty subset of self.data_train
		return DataLoader(
			dataset=self.data_train,
			batch_size=self.hparams.batch_size if not hasattr(self.hparams, "batch_size_train") else self.hparams.batch_size_train,
			num_workers=self.hparams.num_workers,
			pin_memory=self.hparams.pin_memory,
			shuffle=True,
		)

	def val_dataloader(self):
		if isinstance(self.data_val, list):
			return [
				DataLoader(
				dataset=data_val,
				batch_size=self.hparams.batch_size if not hasattr(self.hparams, "batch_size_val") else self.hparams.batch_size_val,
				num_workers=self.hparams.num_workers,
				pin_memory=self.hparams.pin_memory,
				shuffle=False,
			) for data_val in self.data_val
			]
		else:
			return DataLoader(
				dataset=self.data_val,
				batch_size=self.hparams.batch_size,
				num_workers=self.hparams.num_workers,
				pin_memory=self.hparams.pin_memory,
				shuffle=False,
			)

	def test_dataloader(self):
		if isinstance(self.data_test, list):
			return [
				DataLoader(
				dataset=data_test,
				batch_size=self.hparams.batch_size if not hasattr(self.hparams, "batch_size_test") else self.hparams.batch_size_test,
				num_workers=self.hparams.num_workers,
				pin_memory=self.hparams.pin_memory,
				shuffle=False,
			) for data_test in self.data_test
			]
		else:
			return DataLoader(
				dataset=self.data_test,
				batch_size=self.hparams.batch_size,
				num_workers=self.hparams.num_workers,
				pin_memory=self.hparams.pin_memory,
				shuffle=False,
			)

	def teardown(self, stage: Optional[str] = None) -> None:
		"""Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
		`trainer.test()`, and `trainer.predict()`.

		:param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
			Defaults to ``None``.
		"""
		pass

	def state_dict(self) -> Dict[Any, Any]:
		"""Called when saving a checkpoint. Implement to generate and save the datamodule state.

		:return: A dictionary containing the datamodule state that you want to save.
		"""
		return {}

	def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
		"""Called when loading a checkpoint. Implement to reload datamodule state given datamodule
		`state_dict()`.

		:param state_dict: The datamodule state returned by `self.state_dict()`.
		"""
		pass





if __name__ == '__main__':
	print("start...")

	load_kuka("kuka", "/data/models/diffuser/d4rl_dataset/kuka/kuka_dataset/")
	# dataset = EnvDataset("maze2d-umaze-v1", horizon=32, mode="ep_multi_step%5", preprocess_fns="by_env", normalizer="by_env")
	dataset = EnvEpisodeDataset("kuka", horizon=32, mode="ep_multi_step%5", custom_ds_path="/data/models/diffuser/d4rl_dataset/kuka/kuka_dataset/", preprocess_fns="by_env", normalizer="by_env")

	from denoising_diffusion_pytorch.utils.rendering import KukaRenderer
	renderer = KukaRenderer()
	obs_dim = dataset.env.observation_space.shape[0]
	act_dim = dataset.env.action_space.shape[0]
	
	renderer.episodes2img(
		torch.stack([dataset[-1].trajectories[:,act_dim:]],dim=0).cpu().numpy(),
		path="./debug/test1.png"
	)
	# test random
	ep_shape = [4, 10, *dataset.env.observation_space.shape]
	eps = np.random.randn(*ep_shape)
	renderer.episodes2img(
		eps, 
		path="./debug/test2.png"
	)
