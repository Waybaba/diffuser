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


Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')
TransitionBatch = namedtuple('TransitionBatch', 's s_ act')
EpisodeBatch = namedtuple('EpisodeBatch', 'trajectories conditions')
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
		assert "maze" in env or "cheetah" in env or "kuka" in env, "maze envs not supported, since d4rl does not provide terminal"

		### get dataset
		self.env_name = env
		if "kuka" in self.env_name:
			self.env, self.dataset = load_kuka(env, custom_ds_path)
		else:
			self.env = load_environment(env) # TOODO can not use gym.make ?
			if custom_ds_path: self.dataset = self.env.get_dataset(custom_ds_path)
			else: self.dataset = self.env.get_dataset()
			# self.dataset = d4rl.qlearning_dataset(self.env_name)
			# self.dataset.update(self.env.get_dataset())

		### pre_process
		assert preprocess_fns == "by_env", "only support by_env"
		if "maze" in self.env_name: preprocess_fns = ["maze2d_set_terminals"]
		elif "cheetah" in self.env_name: preprocess_fns = []
		elif "kuka" in self.env_name: preprocess_fns = []
		else: raise NotImplementedError("env not supported")
		self.preprocess_fn = get_preprocess_fn(preprocess_fns, self.env_name) # TOODO do not use original function
		self.dataset = self.preprocess_fn(self.dataset)
		
		### remove keys
		KEYS_NEED = ["observations", "actions", "terminals", "timeouts"]
		keys_to_delete = [k for k in self.dataset.keys() if k not in KEYS_NEED]
		for k in keys_to_delete: del self.dataset[k]
		
		### normalize
		assert normalizer == "by_env", "only support by_env"
		if "maze" in self.env_name: normalizer = "LimitsNormalizer"
		elif "cheetah" in self.env_name: normalizer = "GaussianNormalizer"
		elif "kuka" in self.env_name: normalizer = "DebugNormalizer"
		else: raise NotImplementedError("env not supported")
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
		elif "cheetah" in self.env_name:
			from diffuser.utils.rendering import MuJoCoRenderer
			self.renderer = MuJoCoRenderer(self.env_name)
		elif "kuka" in self.env_name:
			from denoising_diffusion_pytorch.utils.rendering import KukaRenderer
			self.renderer = KukaRenderer()
		else:
			raise NotImplementedError("env not supported")

	def __len__(self):
		return len(self.indices)

	def get_episodes_ref(self, ep_num=4):
		""" get reference episodes from dataset
			a list of reference episodes [{
				"s": (T, obs_dim)
				"act": (T, act_dim)
				...
			}]
			always return the first {ep_num} episodes, since we do not distinguish train, val
			ps. return are unnormalized!
		"""
		# if not hasattr(self, "episodes_ref") or len(self.episodes_ref) == ep_num:
		dataset = self.dataset
		episodes_ref = []
		import random
		cur = random.randint(0, len(dataset["terminals"]) - 1)
		for i in range(ep_num):
			start = cur
			while True:
				done = dataset["terminals"][cur]
				if "timeouts" in dataset: done |= dataset["timeouts"][cur]
				cur += 1
				if done: 
					end = cur
					break
			cur_dict = {}
			cur_dict["s"] = dataset["observations"][start:end]
			cur_dict["act"] = dataset["actions"][start:end]
			cur_dict["s_"] = dataset["observations"][start+1:end+1]
			# cur_dict["r"] = dataset["rewards"][start:end]
			for k, v in dataset.items():
				if k.startswith("infos/"): # for ther d4rl keys such as infos/qvel
					cur_dict[k[6:]] = v[start:end]
			episodes_ref.append(cur_dict)
		self.episodes_ref = episodes_ref
		# to cpu if cpu, to numpy if tensor
		for i in range(len(self.episodes_ref)):
			for k, v in self.episodes_ref[i].items():
				if torch.is_tensor(v): self.episodes_ref[i][k] = v.cpu().numpy()
		# unnormalize
		for i in range(len(self.episodes_ref)):
			self.episodes_ref[i]["s"] = self.normalizer.unnormalize(self.episodes_ref[i]["s"], "observations")
			self.episodes_ref[i]["s_"] = self.normalizer.unnormalize(self.episodes_ref[i]["s_"], "observations")
			self.episodes_ref[i]["act"] = self.normalizer.unnormalize(self.episodes_ref[i]["act"], "actions")
		# cut to the same length
		min_len = min([len(ep["s"]) for ep in self.episodes_ref])
		for i in range(len(self.episodes_ref)):
			for k, v in self.episodes_ref[i].items():
				if isinstance(v, np.ndarray): self.episodes_ref[i][k] = v[:min_len]
		return self.episodes_ref

class EnvEpisodeDataset(EnvDataset):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.horizon = kwargs["horizon"]
		self.kwargs = kwargs
		self.indices = self.make_indices(self.dataset)

	def make_indices(self, dataset):
		"""
			makes indices for sampling from dataset;
			each index maps to a datapoint
			(N, 2)
			each element is (start, end)
		"""
		# fast_idx_making = True
		if self.kwargs["mode"] == "default":
			dones = dataset["terminals"]
			if "timeouts" in dataset: dones |= dataset["timeouts"]
			dones_idxes = torch.where(dones)[0]
			indices = []
			start = 0
			print("making indexes ...")
			for i in tqdm(range(len(dones_idxes))):
				if dones_idxes[i] > start: 
					if dones_idxes[i] - start >= self.horizon:
						indices.append([start, dones_idxes[i]])
						if os.environ.get("DEBUG", "false").lower()=="true" and len(indices) > 10000: return torch.tensor(indices)
					start = dones_idxes[i] + 1
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
			start = 0
			max_gap = multi_step * self.horizon
			for start in tqdm(range(0, len(dones) - max_gap)):
				for inter in range(1, int(multi_step) + 1):
					indices.append([start, start + self.horizon * inter, inter])
					if os.environ.get("DEBUG", "false").lower()=="true" and len(indices) > 10000: return torch.tensor(indices)
			indices = torch.tensor(indices)
		elif self.kwargs["mode"].startswith("ep_multi_step"):
			"""
				same as before but is episode based
				would be use indices cross episodes
			"""
			indices = []
			mode, multi_step = self.kwargs["mode"].split("%")
			multi_step = int(multi_step)
			dones = dataset["terminals"]
			if "timeouts" in dataset: dones |= dataset["timeouts"]
			dones_idxes = torch.where(dones)[0]
			
			print("making indexes for episode-based multi-step ...")
			start = 0
			max_gap = multi_step * self.horizon
			for end in tqdm(dones_idxes):
				for i in range(start, end - max_gap + 1):
					for inter in range(1, multi_step + 1):
						indices.append([i, i + self.horizon * inter, inter])
				start = end + 1  # Move to the start of the next episode
				if os.environ.get("DEBUG", "false").lower()=="true" and len(indices) > 10000: 
					return torch.tensor(indices)
				
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
		# 		self.horizon - 1: observations[-1],
		# 	})
		return cond
	
	def __getitem__(self, idx):
		""" 
		"""
		if self.kwargs["mode"] == "default":
			### B random slice
			start, end = self.indices[idx]
			start = np.random.randint(start, end - self.horizon + 1)
			end = start + self.horizon
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
		super().__init__(*args, **kwargs)
		self.indices = self.make_indices(self.dataset, kwargs["multi_step"])

	def make_indices(self, dataset, multi_step):
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
		num_data = len(dataset["observations"])
		dones = dataset.get("terminals", np.zeros(num_data, dtype=bool))
		if "timeouts" in dataset: dones |= dataset["timeouts"]

		valid_indices = torch.where(dones == 0)[0]
		
		indices = []
		print("making indices ...")
		for i in tqdm(valid_indices.cpu().numpy()):
			for j in range(1, multi_step + 1):
				end_idx = i + j
				if dones[end_idx]: 
					indices.append([i, end_idx])
					break
				if end_idx >= num_data: break
				indices.append([i, end_idx])
				if os.environ.get("DEBUG", "false").lower()=="true" and len(indices) > 10000: return np.array(indices)


		print("Dataset make indices done, the length is {}".format(len(indices)))

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
