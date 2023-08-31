from collections import namedtuple
import numpy as np
import torch
from tqdm import tqdm
from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer
from scipy.interpolate import interp1d
import gym
import d4rl
from torch.nn import functional as F 
from pytorch_lightning import LightningDataModule
from typing import Any, Dict, Optional, Tuple
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from diffuser.datasets.normalization import get_normalizer



Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')
FillActBatch = namedtuple('FillActBatch', 's s_ act')

### dataset

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
		

@torch.no_grad()
class SequenceGPUDataset:
	"""
		TODO Now, we use all episode then interpolation to fixed length
		env:
		horizon:
		only_start_end_episode: false for HER
		horizon: length of the trajectory
		normalizer: 'LimitsNormalizer' or 'StandardNormalizer'
			LimitsNormalizer: normalize to [-1, 1]
		preprocess_fns: []
		seed: None
		custom_ds_path: ""
	"""
	def __init__(self, env, horizon, preprocess_fns=[], only_start_end_episode=False, normalizer='LimitsNormalizer', seed=None, custom_ds_path=None):
		assert type(env) == str, "env should be a string"
		assert "maze" in env, "maze envs not supported, since d4rl does not provide terminal"
		assert normalizer == "LimitsNormalizer", "only support LimitsNormalizer"
		assert only_start_end_episode, "only support only_start_end_episode"

		self.horizon = horizon
		self.env_name = env
		
		### get dataset		
		self.env = env = load_environment(env) # ! DEBUG can not use gym.make ?
		if custom_ds_path: self.dataset = env.get_dataset(custom_ds_path)
		else: self.dataset = env.get_dataset()
		# self.dataset = d4rl.qlearning_dataset(env)
		# self.dataset.update(env.get_dataset())

		### pre_process
		assert not preprocess_fns or preprocess_fns[0] == "maze2d_set_terminals"
		self.preprocess_fn = get_preprocess_fn(preprocess_fns, self.env_name)
		self.dataset = self.preprocess_fn(self.dataset)
		
		### remove keys
		KEYS_NEED = ["observations", "actions", "rewards", "terminals"]
		keys_to_delete = [k for k in self.dataset.keys() if k not in KEYS_NEED]
		for k in keys_to_delete: del self.dataset[k]
		
		### normalize
		self.observation_dim = self.dataset['observations'].shape[1]
		self.action_dim = self.dataset['actions'].shape[1]
		self.normalizer = DatasetNormalizerW(self.dataset, normalizer)
		for k in ["observations", "actions"]:
			self.dataset[k] = self.normalizer.normalize(self.dataset[k], k)

		### put into GPU
		for k in self.dataset.keys():
			self.dataset[k] = torch.tensor(self.dataset[k]).cuda()

		### make indexes (all that both terminals and timeouts)
		self.indices = self.make_indices(self.dataset)
	
	def make_indices(self, dataset):
		"""
			makes indices for sampling from dataset;
			each index maps to a datapoint
			(N, 2)
			each element is (start, end)
		"""
		dones = dataset["terminals"]
		starts = (~dones) & torch.cat((torch.tensor([1]).cuda(),dones[:-1])) # 0 and the previous is 1
		ends = dones & (~torch.cat((torch.tensor([1]).cuda(),dones[:-1]))) # 1 and the previous is 0
		starts = torch.where(starts)[0]
		ends = torch.where(ends)[0]
		starts = starts[:-1]
		assert len(starts) == len(ends), "starts and ends should have the same length"
		indices = torch.stack([starts, ends], dim=1)
		return indices

	def get_conditions(self, observations):
		'''
			condition on both the current observation and the last observation in the plan
		'''
		assert "maze" in self.env_name, "use only 0 if not maze"
		return {
			0: observations[0],
			self.horizon - 1: observations[-1],
		}
	
	def __getitem__(self, idx):
		""" TODO 
		"""
		start, end = self.indices[idx]

		observations = self.dataset["observations"][start:end]
		actions = self.dataset["actions"][start:end]

		### interpolation
		observations = observations.T.unsqueeze(0)
		actions = actions.T.unsqueeze(0)
		observations = F.interpolate(observations, size=(self.horizon), mode='linear', align_corners=False)
		actions = F.interpolate(actions, size=(self.horizon), mode='linear', align_corners=False)
		observations = observations.squeeze(0).T
		actions = actions.squeeze(0).T

		conditions = self.get_conditions(observations)
		trajectories = torch.cat([actions, observations], axis=-1)
		batch = Batch(trajectories, conditions)
		return batch

	def __len__(self):
		return len(self.indices)


class SequenceDataset(torch.utils.data.Dataset):

	def __init__(self, env='hopper-medium-replay', horizon=64,
		normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
		max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None,custom_ds_path=None):
		self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
		self.env = env = load_environment(env)
		self.env.seed(seed)
		self.horizon = horizon
		self.use_padding = use_padding
		itr = sequence_dataset(env, self.preprocess_fn, custom_ds_path=custom_ds_path)

		# ! DEBUG adjust buffer size by need
		dataset = env.get_dataset(custom_ds_path)
		lengths = np.where(dataset["terminals"])[0]
		lengths = lengths - np.concatenate([[0], lengths[:-1]])
		print(f'[ datasets ] Dataset size: {len(lengths)}')
		print(f'[ datasets ] Episode length_max: {np.max(lengths)}')
		n_episodes = len(lengths)
		self.max_path_length = max_path_length = np.max(lengths)
		# ! DEBUG

		fields = ReplayBuffer(n_episodes, max_path_length, termination_penalty)
		
		print("\n### add path to buffer ...")
		for i, episode in tqdm(enumerate(itr),total=n_episodes):
			# ! DEBUG set start and end to nearest int
			if len(episode["rewards"]) == 0: continue
			# episode["observations"][0] = np.round(episode["observations"][0])
			# episode["observations"][-1] = np.round(episode["observations"][-1])
			episode["observations"][0] = 0.0
			episode["observations"][-1] = 0.0
			# !
			fields.add_path(episode)
		fields.finalize()


		self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
		self.indices = self.make_indices(fields.path_lengths, horizon)

		self.observation_dim = fields.observations.shape[-1]
		self.action_dim = fields.actions.shape[-1]
		self.fields = fields
		self.n_episodes = fields.n_episodes
		self.path_lengths = fields.path_lengths
		self.normalize()

		print(fields)
		# shapes = {key: val.shape for key, val in self.fields.items()}
		# print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

	def normalize(self, keys=['observations', 'actions']):
		'''
			normalize fields that will be predicted by the diffusion model
		'''
		for key in keys:
			array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
			normed = self.normalizer(array, key)
			self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

	def make_indices(self, path_lengths, horizon):
		'''
			makes indices for sampling from dataset;
			each index maps to a datapoint
		'''
		indices = []
		for i, path_length in enumerate(path_lengths):
			max_start = min(path_length - 1, self.max_path_length - horizon)
			if not self.use_padding:
				max_start = min(max_start, path_length - horizon)
			for start in range(max_start):
				end = start + horizon
				indices.append((i, start, end))
		indices = np.array(indices)
		return indices

	def get_conditions(self, observations):
		'''
			condition on current observation for planning
		'''
		return {0: observations[0]}

	def __len__(self):
		return len(self.indices)

	def __getitem__(self, idx, eps=1e-4):
		path_ind, start, end = self.indices[idx]

		observations = self.fields.normed_observations[path_ind, start:end]
		actions = self.fields.normed_actions[path_ind, start:end]

		conditions = self.get_conditions(observations)
		trajectories = np.concatenate([actions, observations], axis=-1)
		batch = Batch(trajectories, conditions)
		return batch

class FillActDataset(SequenceDataset):
	def __init__(self, env, custom_ds_path=None, multi_step=1):
		"""
		multi_step: how many steps to skip, 1 for only the p(a|s,s'), >1 would be random sample
		"""
		# Create the environment
		assert type(env) == str, "env should be a string"
		assert "maze" not in env, "maze envs not supported, since d4rl does not provide terminal"
		self.env = env = gym.make(env)
		self.dataset = d4rl.qlearning_dataset(env)
		self.dataset.update(env.get_dataset())

		### make indexes (all that both terminals and timeouts)
		self.indices = self.make_indices(self.dataset, multi_step)

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

		valid_indices = np.where(dones == 0)[0]
		
		pairs = []
		print("making indices ...")
		for i in tqdm(valid_indices):
			for j in range(1, multi_step + 1):
				end_idx = i + j
				if dones[end_idx]: 
					pairs.append([i, end_idx])
					break
				if end_idx >= num_data: break
				pairs.append([i, end_idx])
		
		return np.array(pairs)
	
	def __getitem__(self, idx):
		start, end = self.indices[idx]
		s = self.dataset["observations"][start]
		s_ = self.dataset["observations"][end]
		act = self.dataset["actions"][start]
		return FillActBatch(s, s_, act)

	def __len__(self):
		return len(self.indices)

class GoalDataset(SequenceDataset):

	def get_conditions(self, observations):
		'''
			condition on both the current observation and the last observation in the plan
		'''
		return {
			0: observations[0],
			self.horizon - 1: observations[-1],
		}

class WaybabaMaze2dDataset(GoalDataset):
	def make_indices(self, path_lengths, horizon):
		'''
			makes indices for sampling from dataset;
			each index maps to a datapoint
		'''
		indices = []
		for i, path_length in enumerate(path_lengths):
			if path_length > 3:
				start = 0
				end = path_length
				indices.append((i, start, end))
		indices = np.array(indices)
		return indices
	
	def interpolate_data(self, data, old_time, new_time):
		interpolated_data = np.zeros((self.horizon, data.shape[1]), dtype=data.dtype)
		for i in range(data.shape[1]):
			interp_func = interp1d(old_time, data[:, i], kind='linear', fill_value="extrapolate")
			interpolated_data[:, i] = interp_func(new_time)
		return interpolated_data
	
	def __getitem__(self, idx, eps=1e-4):
		""" interpolation to make all the length is equal to horizon
			Batch[0] is the trajectory, [self.horizon, 4]
		"""
		path_ind, start, end = self.indices[idx]

		observations = self.fields.normed_observations[path_ind, start:end] # [T, 4]
		actions = self.fields.normed_actions[path_ind, start:end] # [T, 2]

		# interpolation
		T = observations.shape[0]
		old_time = np.linspace(0, 1, T)
		new_time = np.linspace(0, 1, self.horizon)
		observations = self.interpolate_data(observations, old_time, new_time)
		actions = self.interpolate_data(actions, old_time, new_time)


		conditions = self.get_conditions(observations)
		trajectories = np.concatenate([actions, observations], axis=-1)
		batch = Batch(trajectories, conditions)
		return batch

class ValueDataset(SequenceDataset):
	'''
		adds a value field to the datapoints for training the value function
	'''

	def __init__(self, *args, discount=0.99, normed=False, **kwargs):
		super().__init__(*args, **kwargs)
		self.discount = discount
		self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]
		self.normed = False
		if normed:
			self.vmin, self.vmax = self._get_bounds()
			self.normed = True

	def _get_bounds(self):
		print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
		vmin = np.inf
		vmax = -np.inf
		for i in range(len(self.indices)):
			value = self.__getitem__(i).values.item()
			vmin = min(value, vmin)
			vmax = max(value, vmax)
		print('✓')
		return vmin, vmax

	def normalize_value(self, value):
		## [0, 1]
		normed = (value - self.vmin) / (self.vmax - self.vmin)
		## [-1, 1]
		normed = normed * 2 - 1
		return normed

	def __getitem__(self, idx):
		batch = super().__getitem__(idx)
		path_ind, start, end = self.indices[idx]
		rewards = self.fields['rewards'][path_ind, start:]
		discounts = self.discounts[:len(rewards)]
		value = (discounts * rewards).sum()
		if self.normed:
			value = self.normalize_value(value)
		value = np.array([value], dtype=np.float32)
		value_batch = ValueBatch(*batch, value)
		return value_batch

class AvgCoordinateDataset(SequenceDataset):
	'''
		adds a value field to the datapoints for training the value function
	'''
	COORDINATE = None # 0 for x, 1 for y
	LOWER = None # True for lower, False for higher
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		from diffuser.sampling.guides import NoTrainGuideAvgCoordinate
		self.guide = NoTrainGuideAvgCoordinate()
		self.guide.LOWER = self.LOWER
		self.guide.COORDINATE = self.COORDINATE
	
	def __getitem__(self, idx):
		batch = super().__getitem__(idx)
		value = self.guide.cal_average_coordinate(batch.trajectories)
		value = np.array([value], dtype=np.float32)
		value_batch = ValueBatch(*batch, value)
		return value_batch



### datamodule

class FillActDataModule(LightningDataModule):
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
		import gym
		gym.make(self.hparams.env)

	def	setup_info(self):
		self.info = {
			"obs_dim": self.dataset.env.observation_space.shape[0],
			"act_dim": self.dataset.env.action_space.shape[0],
			"env": self.dataset.env,
			"data_train": self.data_train,
			"data_val": self.data_val,
			"data_test": self.data_test,
			"renderer": self.hparams.renderer
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
			self.dataset = FillActDataset(self.hparams.env, self.hparams.custom_ds_path, self.hparams.multi_step)
			train_val_test_split = self.hparams.train_val_test_split \
				if type(self.hparams.train_val_test_split[0]) != float else \
				[int(len(self.dataset) * split) for split in self.hparams.train_val_test_split]
			train_val_test_split[0] = len(self.dataset) - sum(train_val_test_split[1:])
			self.data_train, self.data_val, self.data_test = random_split(
				dataset=self.dataset,
				lengths=train_val_test_split,
				# generator=torch.Generator().manual_seed(0),
			)
			self.data_val = [self.data_val]
			self.data_test = [self.data_test]
			
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