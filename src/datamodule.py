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
from scipy.interpolate import interp1d
import torch
import os
import random
from copy import deepcopy
import torch.nn.functional as F
import hashlib
from src.func import *
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

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

def load_quickdraw(env_name):
	ds = QuickdrawDataset(data_dir=os.environ['UDATADIR'] + '/quickdraw').get_datadict()
	env = QuickdrawEnv()
	return env, ds

class QuickdrawEnv(gym.Env):
	"""
	A simple Gym environment where the state is equal to the action taken, and the reward is always 0.
	"""

	def __init__(self):
		super().__init__()
		# Define action and observation space
		# They must be gym.spaces objects
		# Example when using discrete actions, you can use spaces.Discrete(N) where N is the action space size
		self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
		self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

	def reset(self):
		"""
		Reset the environment to the initial state.
		"""
		self.state = np.zeros(shape=(3,))
		return self.state

	def step(self, action):
		"""
		Take an action and return the new state, reward, done, and info.
		"""
		self.state = action  # State is updated to be equal to the action
		reward = 0  # Reward is always 0
		# done = False  # In this simple example, we don't have a terminal state
		info = {}  # Additional information, empty in this case
		# action to int
		action = np.array(action, dtype=int)
		if (action == 0).all(): done = True
		else: done = False
		return self.state, reward, done, info

class QuickdrawDataset:
	MODE = "cat"
	def __init__(self, data_dir, use_buf=True):
		self.data_dir = data_dir
		self.use_buf = use_buf

	def make_hidden_stroke(self, start, end, max_dist=5):
		"""
		Generate a hidden stroke between two points
		"""
		x, y = start
		x2, y2 = end
		n =  max(int(np.abs(x - x2) / max_dist) + 1,  int(np.abs(y - y2) / max_dist) + 1)
		dt = 1 / n
		t = np.arange(0, 1 + dt, dt)
		fx = interp1d([0, 1], [x, x2], kind='linear', fill_value="extrapolate")
		fy = interp1d([0, 1], [y, y2], kind='linear', fill_value="extrapolate")
		x_new = fx(t)
		y_new = fy(t)
		# turn to int
		x_new = np.array(x_new, dtype=int)
		y_new = np.array(y_new, dtype=int)
		# to list
		x_new = x_new.tolist()
		y_new = y_new.tolist()
		return (x_new, y_new)

	def load(self):
		import os
		import pickle
		FAST_LOAD = False
		classes = []
		for f_name in os.listdir(self.data_dir):
			if f_name.endswith('.ndjson') and not f_name.startswith('.'): classes.append(f_name[:-7])
			else: pass

		if FAST_LOAD and self.use_buf and os.path.exists(self.data_dir + f'/{self.MODE}_dfs.pkl') and os.path.exists(self.data_dir + f'/{self.MODE}_ds.pkl'):
			with open(self.data_dir + f'/{self.MODE}_dfs.pkl', 'rb') as f:
				self.dfs = pickle.load(f)
			with open(self.data_dir + f'/{self.MODE}_ds.pkl', 'rb') as f:
				self.ds = pickle.load(f)
		else:
			self.dfs = {}
			# for c in classes:
			for c in ["door"]:
				print(f"loading {c}")
				self.dfs[c] = pd.read_json(self.data_dir + '/' + c + '.ndjson', lines=True)

			# df = pd.read_json('/path/to/records.ndjson', lines=True)
			# df.to_json('/path/to/export.ndjson', lines=True)
			"""
			s: (x, y, )
			a: (diff_x, diff_y)
			d: 0 if not end of stroke, 1 if end of stroke, 2 if end of drawing
			"""
			s, done, a = [], [], []
			x, y, h = [], [], []
			for c in self.dfs:
				for i, row in tqdm(self.dfs[c].iterrows()):
					drawing = row['drawing']
					# join all list in drawing
					for i_stroke in range(len(drawing)):
						stroke = drawing[i_stroke]
						x += stroke[0]
						y += stroke[1]
						h += [0] * (len(stroke[0]))
						done += [0] * (len(stroke[0]))
						if i_stroke != len(drawing) - 1:
							hidden_stroke = self.make_hidden_stroke(
								start=(drawing[i_stroke][0][-1],drawing[i_stroke][1][-1]), 
								end=(drawing[i_stroke+1][0][0],drawing[i_stroke+1][1][0]), 
							)
							x += hidden_stroke[0]
							y += hidden_stroke[1]
							h += [1] * (len(hidden_stroke[0]))
							done += [0] * (len(hidden_stroke[0]))
					done[-1] = 1
					# if i > 10: break # ! for DEBUG
			s = [[x[i], y[i], h[i]] for i in range(len(x))]
			s = np.array(s)
			# a = np.diff(s, axis=0)[:,:2]
			# a = np.vstack([a, np.zeros(2)])
			# a[done == 1] = 0
			a = np.vstack([s[1:,:], np.zeros(3)])
			a[done==1] = 0
			r = np.zeros_like(done)
			self.ds = {"observations": s, "actions": a, "terminals": np.array(done), "rewards": r, "timeouts": np.zeros_like(done)}

			self.ds = self.norm_length(self.ds, length=32)
			# self.ds = {"observations": self.ds["observations"], "actions": self.ds["actions"], "terminals": np.array(self.ds["terminals"]), "rewards": self.ds["rewards"], "timeouts": self.ds["timeouts"]}
			if FAST_LOAD:
				with open(self.data_dir + f'/{self.MODE}_dfs.pkl', 'wb') as f:
					pickle.dump(self.dfs, f)
				with open(self.data_dir + f'/{self.MODE}_ds.pkl', 'wb') as f:
					pickle.dump(self.ds, f)
			# ["observations", "actions", "terminals", "timeouts", "rewards"]
			# self.ds_2 = {
			# 	"observations": self.ds["s"],
			# 	"actions": self.ds["a"],
			# 	"terminals": self.ds["d"],
			# 	"timeouts": np.zeros_like(self.ds["d"]),
			# 	"rewards": self.ds["r"],
			# }
		# # drawing and save as ./output/quickdraw/{draw_idx}.png
		# import matplotlib.pyplot as plt
		# import os
		# output_path = "./output" + '/quickdraw'
		# if not os.path.exists(output_path): os.makedirs(output_path)
		# drawing_s_cur = []
		# stroke_s_cur = []
		# plt.figure()
		# plt.gca().invert_yaxis()
		# s, done = self.ds["observations"], self.ds["terminals"]
		# for i in range(len(s)):
		# 	if done[i] == 1:
		# 		plt.savefig(output_path + '/' + str(i) + '.png')
		# 		print(f"saved {output_path}/{i}.png")
		# 		plt.close()
		# 		plt.figure()
		# 	elif s[i,2] == 1:
		# 		if i+2 >= len(s): continue
		# 		plt.plot(s[i:i+2, 0], s[i:i+2, 1], color='red')
		# 	elif s[i,2] == 0: 
		# 		if i+2 >= len(s): continue
		# 		plt.plot(s[i:i+2, 0], s[i:i+2, 1], color='blue')
		# plt.close()
		
		# # TODO revise y axix

	def get_datadict(self):
		if not hasattr(self, 'ds'): self.load()
		return self.ds

	def norm_length(self, ds, length):
		""" interpolation ds to make length equal to `length` for each episode
		1. use "terminals" as indication of episodes
		2. for "observations" keep the start and end the same, interpolate to length
		3. for "actions", at the episode end, action is 0, is the next observation for others
		4. reward and timeouts: all zero
		ds = {
			"observations": s, # (l, obs_dim)
			"actions": a, # (l, act_dim)
			"terminals": np.array(done), # (l,) # 1 when episode ends
			"rewards": r, # (l,)
			"timeouts": np.zeros_like(done) # (l,)
		}
		"""
		new_ds = {
			"observations": [],
			"actions": [],
			"terminals": [],
			"rewards": [],
			"timeouts": []
		}

		# Identifying episode boundaries
		episode_boundaries = np.where(ds['terminals'] == 1)[0]
		start_idx = 0

		for end_idx in episode_boundaries:
			# Interpolating observations and actions
			episode_data = ds['observations'][start_idx:end_idx + 1]
			old_time_steps = np.linspace(0, 1, len(episode_data))
			new_time_steps = np.linspace(0, 1, length)

			# Using linear interpolation
			interpolator = interp1d(old_time_steps, episode_data, axis=0)
			new_data = interpolator(new_time_steps)

			# make the [-1] element round to 0,1
			new_data = np.round(new_data)

			# Handling terminals, rewards, and timeouts
			new_ds['observations'].append(new_data)
			new_ds['actions'].append(np.concatenate([new_data[1:], np.zeros((1,ds['actions'].shape[1]))], axis=0))
			new_ds['terminals'].append(np.array([0]*(length-1) + [1]))
			new_ds['rewards'].append(np.zeros(length))
			new_ds['timeouts'].append(np.zeros(length, dtype=np.int32))

			start_idx = end_idx + 1

		# Converting lists to numpy arrays
		for key in new_ds:
			new_ds[key] = np.concatenate(new_ds[key], axis=0)

		return new_ds



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
		assert [env.startswith(v) for v in ["quickdraw", "minari:","maze", "walker2d", "hopper", "halfcheetah", "reacher", "kitchen", "hammer", "door", "pen", "relocate"]].count(True) == 1, f"env {env} not supported"

		### get dataset (setup self.dataset, self.env)
		self.env_name = env
		if "kuka" in self.env_name:
			self.env, self.dataset = load_kuka(self.env_name, custom_ds_path)
		elif [self.env_name.startswith(v) for v in ["hammer","door", "relocate","pen", "kitchen"]].count(True) == 1:
			self.env, self.dataset = load_minari(self.env_name)
		elif self.env_name.startswith("minari:"):
			self.env, self.dataset = load_custom_minari(self.env_name.split(":")[1])
		elif [self.env_name.endswith(suf) for suf in ["mixed", "random-expert"]].count(True) == 1:
			# e.g. halfcheetah-mixed -> use all, halfcheetah-random-expert -> use random and expert
			if self.env_name.endswith("mixed"):
				ds_suffix_list = ["random", "medium", "expert"]
			elif self.env_name.endswith("random-expert"):
				ds_suffix_list = ["random", "expert"]
			base_name = self.env_name.split("-")[0]
			ds_list = []
			for suffix in ds_suffix_list:
				env_name_ = base_name + "-" + suffix + "-v2"
				env_ = load_environment(env_name_)
				ds = env_.get_dataset()
				ds_list.append(ds)
			self.env, self.dataset = env_, {}
			for k in ds_list[0].keys():
				self.dataset[k] = np.concatenate([ds[k] for ds in ds_list], axis=0)
		elif self.env_name.startswith("quickdraw"):
			self.env, self.dataset = load_quickdraw(self.env_name)
		else:
			self.env = load_environment(self.env_name) # TOODO can not use gym.make ?
			if custom_ds_path: self.dataset = self.env.get_dataset(custom_ds_path)
			else: self.dataset = self.env.get_dataset()
			# self.dataset = d4rl.qlearning_dataset(self.env_name)
			# self.dataset.update(self.env.get_dataset())

		### pre_process fns
		assert preprocess_fns == "by_env", "only support by_env"
		if "maze" in self.env_name: preprocess_fns = ["maze2d_set_terminals"]
		elif [self.env_name.startswith(v) for v in ["halfcheetah", "walker2d", "hopper", "hammer","door", "relocate","pen", "kitchen"]].count(True) == 1: 
			preprocess_fns = []
		elif self.env_name.startswith("minari:"):
			preprocess_fns = []
		elif "kuka" in self.env_name: 
			preprocess_fns = []
		elif self.env_name.startswith("quickdraw"):
			preprocess_fns = []
		else: raise NotImplementedError("env not supported")
		self.preprocess_fn = get_preprocess_fn(preprocess_fns, self.env_name) # TOODO do not use original function
		self.dataset = self.preprocess_fn(self.dataset)
		
		### remove keys
		KEYS_NEED = ["observations", "actions", "terminals", "timeouts", "rewards"]
		KEYS_NEED += ["infos/qpos", "infos/qvel", "infos/action_lop_probs"] # TOODO for controller reset position
		keys_to_delete = [k for k in self.dataset.keys() if k not in KEYS_NEED]
		for k in keys_to_delete: del self.dataset[k]
		
		### normalize
		if normalizer == "by_env":
			if "maze" in self.env_name: 
				normalizer = "LimitsNormalizer"
			elif [self.env_name.startswith(v) for v in ["halfcheetah", "walker2d", "hopper"]].count(True) == 1: 
				normalizer = "GaussianNormalizer" # DebugNormalizer, GaussianNormalizer
				# normalizer = "DebugNormalizer" # ! Jan 9 
			elif [self.env_name.startswith(v) for v in ["hammer","door", "relocate","pen", "kitchen"]].count(True) == 1:
				normalizer = "DebugNormalizer"
			elif self.env_name.startswith("minari:"):
				normalizer = "DebugNormalizer"
			elif "kuka" in self.env_name: normalizer = "LimitsNormalizer"
			elif self.env_name.startswith("quickdraw"): 
				normalizer = "GaussianNormalizer"
			else: raise NotImplementedError(f"env {self.env_name} not supported")
		else:
			normalizer = normalizer
		self.observation_dim = self.dataset['observations'].shape[1]
		self.action_dim = self.dataset['actions'].shape[1]
		self.normalizer = DatasetNormalizerW(self.dataset, normalizer)
		for k in ["observations", "actions"]:
			# if [self.env_name.startswith(v) for v in ["hammer","door", "relocate","pen", "kitchen"]].count(True) == 1:
			# 	if k == "actions": continue # ! for DEBUG
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
		elif [self.env_name.startswith(v) for v in ["hammer","door", "relocate","pen", "kitchen"]].count(True) == 1:
			from diffuser.utils.rendering import MarinaRenderer
			self.renderer = MarinaRenderer(self.env_name)
		elif self.env_name.startswith("minari:"):
			from diffuser.utils.rendering import MuJoCoRenderer
			self.renderer = MuJoCoRenderer(self.env_name)
		elif self.env_name.startswith("quickdraw"): 
			from diffuser.utils.rendering import QuickdrawRenderer
			self.renderer = QuickdrawRenderer()
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
		# remove consecutive ones
		tmp = []
		for i in range(len(termination_indices)):
			if i == 0 or termination_indices[i] != termination_indices[i-1] + 1:
				tmp.append(termination_indices[i])
		termination_indices = torch.tensor(tmp)
		
		# Randomly select {num_episodes} starting indices for the episodes
		random_start_indices = np.random.randint(0, len(termination_indices) - 3, num_episodes)
		episode_boundaries = torch.stack([termination_indices[random_start_indices]+1,
										termination_indices[random_start_indices + 1]], dim=1)
		
		# Construct the list of reference episodes
		reference_episodes = []
		for start_idx, end_idx in episode_boundaries:
			episode_data = {
				"s": dataset["observations"][start_idx:end_idx-1],
				"act": dataset["actions"][start_idx:end_idx-1],
				"s_": dataset["observations"][start_idx + 1:end_idx -1 + 1],
				"r": dataset["rewards"][start_idx:end_idx-1],
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
		DEBUG_MODE = os.environ.get("DEBUG", "false").lower()=="true"
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
						if DEBUG_MODE and len(indices) > 10000: return torch.tensor(indices)
					ep_start = dones_idxes[i] + 1
			indices = torch.tensor(indices)
		elif self.kwargs["mode"] == "special%maze":
			dones = dataset["terminals"]
			if "timeouts" in dataset: dones |= dataset["timeouts"]
			dones_idxes = torch.where(dones)[0]
			MIN, MAX, INTER = 20, 200, 5
			lengths = list(range(MIN, MAX, INTER))
			indices = []
			for i_start in tqdm(range(len(dones)-MAX-1)):
				for l in lengths:
					indices.append([i_start, i_start + l])
					if DEBUG_MODE and len(indices) > 10000: return torch.tensor(indices)
			indices = torch.tensor(indices)
		elif self.kwargs["mode"].startswith("interpolation"):
			"""
				indices: [(ep_start, ep_end)]
			"""
			dones = dataset["terminals"]
			if "timeouts" in dataset: dones |= dataset["timeouts"]
			dones_idxes = torch.where(dones)[0]
			indices = []
			ep_start = 0
			print("making indexes ...")
			for i in tqdm(range(len(dones_idxes))):
				if dones_idxes[i] > ep_start: 
					indices.append([ep_start, dones_idxes[i]])
					if DEBUG_MODE and len(indices) > 10000: return torch.tensor(indices)
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
					if DEBUG_MODE and len(indices) > 10000: return torch.tensor(indices)
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
				if DEBUG_MODE and len(indices) > 10000: return torch.tensor(indices)
				
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
			full_len = len(dataset["terminals"])
			
			print("making indexes for valid episode-based multi-step ...")
			ep_start = 0
			for ep_end in tqdm(dones_idxes):
				for i in range(ep_start, ep_end): # 101 200=doneTrue i=101
					for inter in range(1, multi_step + 1):
						item_end = i + self.kwargs["horizon"] * inter # 200+101
						invalid_start = ((ep_end-i) // inter) # 100
						if item_end < full_len:
							# 101, 301, 1, 100
							indices.append([i, item_end, inter, invalid_start])
				ep_start = ep_end + 1  # Move to the start of the next episode
				if DEBUG_MODE and len(indices) > 10000: return torch.tensor(indices)
			
			indices = torch.tensor(indices)
		elif self.kwargs["mode"].startswith("valid_multi_step_epstart"):
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
			full_len = len(dataset["terminals"])
			
			print("making indexes for valid episode-based multi-step ...")
			ep_start = 0
			for ep_end in tqdm(dones_idxes):
				if ep_end - ep_start < self.kwargs["horizon"]: continue # ! this would miss a lot of data
				for i in [ep_start]: # 101 200=doneTrue i=101
					for inter in range(1, multi_step + 1):
						item_end = i + self.kwargs["horizon"] * inter # 200+101
						invalid_start = ((ep_end-i) // inter) # 100
						if item_end < full_len:
							# 101, 301, 1, 100
							indices.append([i, item_end, inter, invalid_start])
				ep_start = ep_end + 1  # Move to the start of the next episode
				if DEBUG_MODE and len(indices) > 10000: return torch.tensor(indices)
			
			indices = torch.tensor(indices)
		else:
			raise NotImplementedError("mode not supported")

		return indices

	def get_conditions(self, observations):
		'''
			condition on both the current observation and the last observation in the plan
		'''
		cond = {0: observations[0]}
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
		elif self.kwargs["mode"].startswith("interpolation"):
			"""
			interpolation to make length == horizon
			"""
			start, end = self.indices[idx]
			observations = self.dataset["observations"][start:end] # (T, obs_dim)
			actions = self.dataset["actions"][start:end]
			T = observations.shape[0]
			observations = self.interpolate_data(observations, self.kwargs["horizon"])
			actions = self.interpolate_data(actions, self.kwargs["horizon"])
			conditions = self.get_conditions(observations)
			trajectories = torch.cat([actions, observations], axis=-1)
			batch = EpisodeBatch(trajectories, conditions)
		elif self.kwargs["mode"].startswith("special%maze"):
			"""
			interpolation to make length == horizon
			"""
			start, end = self.indices[idx]
			observations = self.dataset["observations"][start:end] # (T, obs_dim)
			# turn the last 2 dim of obs to zero
			# observations[:, -2:] = 0. # ! Turn vel to zero for diverse predict
			actions = self.dataset["actions"][start:end]
			T = observations.shape[0]
			observations = self.interpolate_data(observations, self.kwargs["horizon"])
			actions = self.interpolate_data(actions, self.kwargs["horizon"])
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
			assert observations.shape[0] == self.kwargs["horizon"], f"Invalid horizon The related information is {start, end, inter, invalid_start, self.kwargs['horizon']}"
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

	def interpolate_data(self, data, T):
		"""
		data: (T_, obs_dim)
		B, C, W, H
		"""
		res = data.unsqueeze(0) # (1, T_, obs_dim)
		res = res.permute(0,2,1) # (1, obs_dim, T_)
		res = F.interpolate(res, size=(T,), mode="linear", align_corners=False) # (1, obs_dim, T)
		res = res.squeeze(0) # (obs_dim, T)
		res = res.permute(1,0) # (T, obs_dim)
		return res
		
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
			! do not cross episodes
			for each episode
				for each step in [ep_start, ep_end)
					for each interval in [1, min(multi_step, steps_to_end_of_this_episodes)]
						indices append (step, step+interval)
			return indices
		"""
		DEBUG_MODE = os.environ.get("DEBUG", "false").lower()=="true"
		dataset, multi_step = self.dataset, self.kwargs["multi_step"]

		num_data = len(dataset["observations"])
		dones = dataset.get("terminals", np.zeros(num_data, dtype=bool))
		if "timeouts" in dataset: dones |= dataset["timeouts"]

		ep_done_idxes = torch.where(dones == 1)[0]
		ep_start_idxes = torch.cat([torch.tensor([0]).to(ep_done_idxes.device), ep_done_idxes[:-1] + 1])

		indices = []
		print("making indices ...")
		for ep_start, ep_end in tqdm(zip(ep_start_idxes, ep_done_idxes), total=len(ep_start_idxes)):
			for i in range(ep_start, ep_end):
				steps_to_end_of_this_episode = ep_end - i
				for interval in range(1, min(multi_step, steps_to_end_of_this_episode) + 1):
					indices.append([i, i + interval])
					if DEBUG_MODE and len(indices) > 10000: return torch.tensor(indices)
		indices = torch.tensor(indices)
		return indices
	
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
