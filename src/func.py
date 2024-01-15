
from gym.envs.registration import register
from copy import deepcopy
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
import hydra
import os
from tqdm import tqdm
import torch
import wandb
from diffuser.sampling.guides import DummyGuide
from diffuser.sampling.policies import GuidedPolicy
from diffuser.sampling import n_step_guided_p_sample_freedom_timetravel, n_step_guided_p_sample
from collections import namedtuple
import gymnasium as gym

"""names"""
Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')
TransitionBatch = namedtuple('TransitionBatch', 's s_ act')
EpisodeBatch = namedtuple('EpisodeBatch', 'trajectories conditions')
EpisodeValidBatch = namedtuple('EpisodeValidBatch', 'trajectories conditions valids')
MUJOCO_ENVS = ["hopper", "walker2d", "halfcheetah"]


"""Functions"""
import gymnasium as gym

class MergeObsActWrapper(gym.Wrapper):
	""" merge all observation into a single observation
	original observation space: {
		'achieved_goal': Box(-10., 10., (3,), float32),
		'desired_goal': Box(-10., 10., (3,), float32),
		'observation': Box(-inf, inf, (25,), float32)
		...
	}
	new observation space: Box([-10., -10., ..., -inf, -inf], [10., 10., ..., inf, inf], (31,), float32)
	"""
	def __init__(self, env):
		super().__init__(env)
		self.observation_space = gym.spaces.Box(
			low=np.concatenate([env.observation_space[key].low for key in env.observation_space.spaces.keys()]),
			high=np.concatenate([env.observation_space[key].high for key in env.observation_space.spaces.keys()]),
			dtype=np.float32
		)

	def reset(self):
		res = self.env.reset()
		if isinstance(res, tuple): obs, info = res
		else: obs, info = res, {}
		obs = self.merge_obs(obs)
		if isinstance(res, tuple):
			return obs, info
		else:
			return obs

	def step(self, action):
		res = self.env.step(action)
		if len(res) == 4:
			obs, reward, done, info = res
			truncated = False
		elif len(res) == 5:
			obs, reward, done, truncated, info = res
		else:
			raise ValueError("Invalid return value from env.step()")
		obs = self.merge_obs(obs)
		# return
		if len(res) == 4:
			return obs, reward, done, info
		elif len(res) == 5:
			return obs, reward, done, truncated, info
	
	def merge_obs(self, obs):
		return np.concatenate([obs[key] for key in obs.keys()])

class ObsHander:
	def __init__(self):
		self.data = []
	
	def register(self, name, dim_num, space):
		""" space can be a float
		"""
		if isinstance(space, float):
			space = gym.spaces.Box(low=-space, high=space, shape=(dim_num,), dtype=np.float32)
		self.data.append((name, dim_num, space))
	
	def distill(self, obs, name):
		"""
		obs: (all_dim, ) # where all_dim=sum([dim_num for _, dim_num, _ in self.data])
		name: str
		"""
		if name == "goal":
			return self.distill(obs, "base")[...,-3:]
		elif name == "endpoint":
			return self.distill(obs, "base")[...,:3]
		elif name == "achieved":
			return self.distill(obs, "base")[...,-6:-3]
		else:
			idx = 0
			for name_, dim_num, _ in self.data:
				if name_ == name:
					return obs[...,idx:idx+dim_num]
				else:
					idx += dim_num
			raise ValueError("Invalid name {}".format(name))

	def get_all_space(self):
		# all_spc = gym.spaces.Box(
		#     low=np.concatenate([spc.low for _, _, spc in self.data]),
		#     high=np.concatenate([spc.high for _, _, spc in self.data]),
		#     dtype=np.float32
		# )
		# ! use inf since there are bug of that the obs is out of range from panda-gym
		all_spc = gym.spaces.Box(
			low=np.full(sum([dim_num for _, dim_num, _ in self.data]), -np.inf, dtype=np.float32),
			high=np.full(sum([dim_num for _, dim_num, _ in self.data]), np.inf, dtype=np.float32),
			dtype=np.float32
		)
		return all_spc
	
	def rm_goal(self):
		# find base index then minus 3
		base_idx = 0
		for i in range(len(self.data)):
			if self.data[i][0] == "base":
				base_idx = i
				break
		self.data[base_idx] = ("base", self.data[base_idx][1]-3, self.data[base_idx][2])

class PandaAddStateWrapper(gym.Wrapper):
	"""
	note this only for rendering since it only set the joint angle
	without updating the velocity
	"""
	def __init__(self, env):
		super().__init__(env)
		self.obs_handler = ObsHander()
		self.obs_handler.register("base", dim_num=env.observation_space.shape[0], space=env.observation_space)
		
		self.joint_dim = len(env.robot.joint_indices)
		self.obs_handler.register("joint", dim_num=self.joint_dim, space=10.)
		
		if "Push" in self.env.spec.id:
			self.push_obj_dim = 3 # ! only add position, since the orientation is not very useful for rendering and would cost space
			self.obs_handler.register("push_obj", dim_num=self.push_obj_dim, space=10.)
		
		self.observation_space = self.obs_handler.get_all_space()

	def reset(self):
		res = self.env.reset()
		if isinstance(res, tuple): obs, info = res
		else: obs, info = res, {}
		obs = self.append_extra_obs(obs)
		if isinstance(res, tuple):
			return obs, info
		else:
			return obs

	def step(self, action):
		res = self.env.step(action)
		if len(res) == 4:
			obs, reward, done, info = res
			truncated = False
		elif len(res) == 5:
			obs, reward, done, truncated, info = res
		else:
			raise ValueError("Invalid return value from env.step()")
		obs = self.append_extra_obs(obs)
		# return
		if len(res) == 4:
			return obs, reward, done, info
		elif len(res) == 5:
			return obs, reward, done, truncated, info
	
	def get_extra_obs(self):
		obs_ = np.array([
			self.robot.get_joint_angle(joint) \
			for joint in self.robot.joint_indices
		], dtype=np.float32)
		if "Push" in self.env.spec.id:
			tmp = self.env.task.sim.get_base_position("object").astype(np.float32)
			obs_ = np.concatenate([obs_, tmp])
		return obs_

	def append_extra_obs(self, obs):
		return np.concatenate([obs, self.get_extra_obs()])

	def set_state(self, obs):
		self.robot.set_joint_angles(self.obs_handler.distill(obs, "joint"))
		self.sim.set_base_pose("target", self.obs_handler.distill(obs, "base")[-3:], np.array([0.0, 0.0, 0.0, 1.0]))
		if "Push" in self.env.spec.id:
			self.sim.set_base_pose("object", self.obs_handler.distill(obs, "push_obj"), np.array([0.0, 0.0, 0.0, 1.0]))

class PandaNogoalWrapper(gym.Wrapper):
	""" merge all observation into a single observation
	original observation space: {
		'achieved_goal': Box(-10., 10., (3,), float32),
		'desired_goal': Box(-10., 10., (3,), float32),
		'observation': Box(-inf, inf, (25,), float32)
		...
	}
	new observation space: Box([-10., -10., ..., -inf, -inf], [10., 10., ..., inf, inf], (31,), float32)
	"""
	NOGOAL = True
	def __init__(self, env):
		super().__init__(env)
		# ! TODO hard code
		GOAL_DIM = {
			"Reach": (9, 12),
			"Push": (21, 24),
			# TODO
			# TODO
		}
		self.goal_dim = next(dim_ for name, dim_ in GOAL_DIM.items() if name in env.spec.id)
		
		self.observation_space = gym.spaces.Box(
			low=self.rm_dims(env.observation_space.low),
			high=self.rm_dims(env.observation_space.high),
			dtype=np.float32
		)

		self.obs_handler.rm_goal()

	def reset(self):
		res = self.env.reset()
		if isinstance(res, tuple): obs, info = res
		else: obs, info = res, {}
		obs = self.merge_obs(obs)
		if isinstance(res, tuple):
			return obs, info
		else:
			return obs

	def step(self, action):
		res = self.env.step(action)
		if len(res) == 4:
			obs, reward, done, info = res
			truncated = False
		elif len(res) == 5:
			obs, reward, done, truncated, info = res
		else:
			raise ValueError("Invalid return value from env.step()")
		obs = self.merge_obs(obs)
		# return
		if len(res) == 4:
			return obs, reward, done, info
		elif len(res) == 5:
			return obs, reward, done, truncated, info
	
	def merge_obs(self, obs):
		return self.rm_dims(obs)

	def rm_dims(self, in_):
		if len(in_.shape) == 1:
			return np.concatenate([in_[:self.goal_dim[0]], in_[self.goal_dim[1]:]])
		elif len(in_.shape) == 2:
			return np.concatenate([in_[:,:self.goal_dim[0]], in_[:,self.goal_dim[1]:]], axis=1)
		else:
			raise ValueError("Invalid shape {}".format(in_.shape))

	def set_state(self, obs):
		self.env.set_state(obs)
		self.sim.set_base_pose("target", np.array([0.,0.,0.]), np.array([0.0, 0.0, 0.0, 1.0]))

	def distill(self, obs, name):
		return self.obs_handler.distill(obs, name)

def gym_make_panda(env_name):
	# e.g. pandareachdense-sac_10000
	import panda_gym
	e_name = env_name.split("-")[0]
	e_name = e_name[5:][:-5]
	e_name = e_name[0].upper() + e_name[1:]
	e_name = f"Panda{e_name}"
	e_name += "" if "dense" not in env_name else "Dense"
	e_name += "-v3"
	env = gym.make(e_name, render_mode="rgb_array")
	env = MergeObsActWrapper(env)
	env = PandaAddStateWrapper(env)
	if env_name.endswith("nogoal"):
		env = PandaNogoalWrapper(env)
	return env

def load_panda(env_name):
	def dataset_rm_goal(dataset, env):
		dataset["observations"] = env.rm_dims(dataset["observations"])
		return dataset
	import minari
	import gymnasium as gym
	env = gym_make_panda(env_name)
	if env_name.endswith("nogoal"):
		dataset = minari.load_dataset(env_name[:-7])
		dataset = minari_to_d4rl(dataset)
		dataset = dataset_rm_goal(dataset, env)
	else:
		dataset = minari.load_dataset(env_name)
		dataset = minari_to_d4rl(dataset)
	return env, dataset

def wandb_media_wrapper(media):
	"""
	handle two types of rendering result, both to wandb log objects
	"""
	if len(media.shape) == 4: # T, Ch, H, W
		assert media.shape[1] in [3, 4], "Ch number wrong, shape is {media.shape}"
		return wandb.Video(media)
	elif len(media.shape) == 3: # H, W, C
		assert media.shape[2] in [3, 4], "Ch number wrong, shape is {media.shape}"
		return wandb.Image(media)

def load_diffuser(dir_, epoch_):
	print("\n\n\n### loading diffuser ...")
	from src.modelmodule import DiffuserModule
	diffuser_cfg = OmegaConf.load(Path(dir_)/"hydra_config.yaml")
	assert "DiffuserModule" in diffuser_cfg.modelmodule._target_, f"Load config of DiffuserModule with error target {diffuser_cfg.modelmodule._target_}"
	datamodule = hydra.utils.instantiate(diffuser_cfg.datamodule)()
	modelmodule = DiffuserModule.load_from_checkpoint(
		Path(dir_)/"checkpoints"/f"{epoch_}.ckpt",
		dataset_info=datamodule.info,
	)
	return modelmodule

def load_controller(dir_, epoch_):
	print("\n\n\n### loading controller ...")
	from src.modelmodule import FillActModelModule
	diffuser_cfg = OmegaConf.load(Path(dir_)/"hydra_config.yaml")
	assert "FillActModelModule" in diffuser_cfg.modelmodule._target_, f"Load config of FillActModelModule with error target {diffuser_cfg.modelmodule._target_}"
	datamodule = hydra.utils.instantiate(diffuser_cfg.datamodule)()
	modelmodule = FillActModelModule.load_from_checkpoint(
		Path(dir_)/"checkpoints"/f"{epoch_}.ckpt",
		dataset_info=datamodule.info,
	)
	return modelmodule

def full_rollout_once(
		env, 
		planner, 
		actor=None, 
		normalizer_actor=None, 
		plan_freq=1,
		len_max=1000
	):
	"""
	planner: 
		call planner(cond, batch_size=1,verbose=False) and return actions, samples
	actor:
		call actor(obs, obs_, batch_size=1, verbose=False) and return act
	"""
		
	def make_act(actor, history, plan, t_madeplan, normalizer_actor):
		"""
		actor: would generate act, different for diff methods
		history: [obs_dim]*t_cur # note the length should be t_cur so that plan would be made
		"""
		s = history[-1]
		s_ = plan[len(history)-1-t_madeplan+1] # e.g. for first step, len(history)=1, t_madeplan=0, we should use first element of plan as s_
		model = actor
		device = next(actor.parameters()).device
		model.to(device)
		act = model(torch.cat([
			torch.tensor(normalizer_actor.normalize(
				s,
				"observations"
			)).to(device), 
			torch.tensor(normalizer_actor.normalize(
				s_,
				"observations"
			)).to(device)
		], dim=-1).float().to(device))
		act = act.detach().cpu().numpy()
		act = normalizer_actor.unnormalize(act, "actions")
		return act

	def make_plan(planner, history):
		"""
		TODO: use history in guide
		"""
		cond = {0: history[-1]}
		action, samples = planner(cond, batch_size=1,verbose=False)
		plan = samples.observations[0] # (T, obs_dim)
		return plan


	# assert actor.horizon >= plan_freq, "plan_freq should be smaller than horizon"
	assert actor.training == False, "actor should be in eval mode"
	# assert planner.training == False, "planner should be in eval mode"
	print(f"Start full rollout, plan_freq={plan_freq}, len_max={len_max} ...")
	res = {"act": [],"s": [],"s_": [],"r": [],}
	env_step = 0

	t_madeplan = -99999
	
	s = env.reset()
	s = s[0] if isinstance(s, tuple) and len(s)==2 else s # for kuka env
	while True: 
		if env_step - t_madeplan >= plan_freq: # note the max value is horizon - 1 instead of horizon, since the first step is current
			plan = make_plan(planner, res["s"]+[s]) # (horizon, obs_dim)
			t_madeplan = env_step
		a = make_act(actor, res["s"]+[s], plan, t_madeplan, normalizer_actor)
		env_res = env.step(a)
		if len(env_res) == 4: s_, r, done, info = env_res
		elif len(env_res) == 5: 
			s_, r, terminal, timeout, info = env_res
			done = terminal or timeout
		s = s_
		
		res["act"].append(a)
		res["s"].append(s)
		res["s_"].append(s_)
		res["r"].append(r)
		env_step += 1
		if done or env_step > len_max: break
	
	# stack
	for k in res.keys():
		res[k] = np.stack(res[k], axis=0)
	
	print(f"Full Rollout: len={len(res['act'])} reward_sum={sum(res['r'])}")
	return res

def full_rollout_once_witha(
		env, 
		planner, 
		normalizer_actor, 
		plan_freq=1,
		len_max=1000
	):
	"""
	planner: 
		call planner(cond, batch_size=1,verbose=False) and return actions, samples
	actor:
		call actor(obs, obs_, batch_size=1, verbose=False) and return act
	"""

	def make_plan_and_act(planner, history):
		cond = {0: history[-1]}
		action, samples = planner(cond, batch_size=1,verbose=False)
		plan = samples.observations[0] # (T, obs_dim)
		return plan, action # [T, obs_dim], [act_dim]

	assert planner.diffusion_model.training == False, "planner should be in eval mode"
	print(f"Start full rollout, plan_freq={plan_freq}, len_max={len_max} ...")
	res = {"act": [],"s": [],"s_": [],"r": [],}
	env_step = 0
	
	s = env.reset()
	s = s[0] if isinstance(s, tuple) and len(s)==2 else s # for kuka env
	while True: 
		# if env_step - t_madeplan >= plan_freq: # note the max value is horizon - 1 instead of horizon, since the first step is current
		# 	plan, plan_act = make_plan(planner, res["s"]+[s]) # (horizon, obs_dim)
		# 	t_madeplan = env_step
		# a = make_act(res["s"]+[s], plan, plan_act, t_madeplan, normalizer_actor)
		# a = plan_act[0]
		plan, a = make_plan_and_act(planner, res["s"]+[s])
		
		# ! June 9: previous manually worked
		# conditions = {0: torch.tensor(s).to("cuda")}
		# conditions = {k: planner.preprocess_fn(v) for k, v in conditions.items()}
		# samples = planner.diffusion_model(conditions, guide=planner.guide, verbose=False, **planner.sample_kwargs) # ! sample kwargs
		# a = samples.trajectories[0, 0, :planner.action_dim]
		# a = a.cpu().numpy

		env_res = env.step(a)
		if len(env_res) == 4: s_, r, done, info = env_res
		elif len(env_res) == 5: 
			s_, r, terminal, timeout, info = env_res
			done = terminal or timeout
		s = s_
		
		res["act"].append(a)
		res["s"].append(s)
		res["s_"].append(s_)
		res["r"].append(r)
		env_step += 1
		if done or env_step > len_max: break
	
	# stack
	for k in res.keys(): res[k] = np.stack(res[k], axis=0)
	
	print(f"Full Rollout: len={len(res['act'])} reward_sum={sum(res['r'])}")
	return res

def full_rollout_once_bc(
		env, 
		model, 
		normalizer_actor, 
		len_max=1000
	):
	"""
	planner: 
		call planner(cond, batch_size=1,verbose=False) and return actions, samples
	actor:
		call actor(obs, obs_, batch_size=1, verbose=False) and return act
	"""
		
	def make_act(s, normalizer_actor):
		stensor = normalizer_actor.normalize(s, "observations")
		stensor = torch.tensor(stensor).to(next(model.parameters()).device)
		# s to same type as model
		stensor = stensor.float()
		act = model(stensor)
		# 
		act = act.detach().cpu().numpy()
		act = normalizer_actor.unnormalize(act, "actions")
		return act

	# assert planner.training == False, "planner should be in eval mode"
	res = {"act": [],"s": [],"s_": [],"r": [],}
	env_step = 0

	t_madeplan = -99999
	
	s = env.reset()
	s = s[0] if isinstance(s, tuple) and len(s)==2 else s # for kuka env
	while True: 
		a = make_act(s, normalizer_actor)
		env_res = env.step(a)
		if len(env_res) == 4: s_, r, done, info = env_res
		elif len(env_res) == 5: 
			s_, r, terminal, timeout, info = env_res
			done = terminal or timeout
		s = s_
		res["act"].append(a)
		res["s"].append(s)
		res["s_"].append(s_)
		res["r"].append(r)
		env_step += 1
		if done or env_step > len_max: break
	
	# stack
	for k in res.keys():
		res[k] = np.stack(res[k], axis=0)
	
	print(f"Full Rollout: len={len(res['act'])} reward_sum={sum(res['r'])}")
	return res

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
		"terminals": terminals,
		"rewards": np.zeros_like(terminals),
	}
	return env, dataset

def minari_to_d4rl(dataset):
	# ["observations", "actions", "terminals", "timeouts", "rewards"]
	ep_list = [ep for ep in dataset.iterate_episodes()]
	res = {}
	if isinstance(ep_list[0].observations, dict):
		res["observations"] = np.concatenate([ep.observations["observation"] for ep in ep_list], axis=0)
	else:
		res["observations"] = np.concatenate([ep.observations for ep in ep_list], axis=0)
	res["actions"] = np.concatenate([ep.actions for ep in ep_list], axis=0)
	res["terminals"] = np.concatenate([ep.terminations for ep in ep_list], axis=0)
	res["timeouts"] = np.concatenate([ep.truncations for ep in ep_list], axis=0)
	res["rewards"] = np.concatenate([ep.rewards for ep in ep_list], axis=0)
	return res

def load_minari(env):
	import minari
	import gymnasium as gym
	minari.download_dataset(env)
	dataset = minari.load_dataset(env)
	env = dataset.recover_environment()
	dataset = minari_to_d4rl(dataset)
	return env, dataset

def load_custom_minari(env_name):
	import minari
	import gymnasium as gym
	dataset = minari.load_dataset(env_name)
	env = dataset.recover_environment()
	dataset = minari_to_d4rl(dataset)
	return env, dataset

def gen_with_same_cond(policy, episodes_ds):
	"""
	"""
	# get conditions
	episodes_ds_ = deepcopy(episodes_ds)
	res = []
	for i in range(len(episodes_ds_)):
		ep_i = episodes_ds_[i]
		cond = {
			0: episodes_ds_[i]["s"][0]
		}
		del ep_i["act"] # to avoid misuse
		del ep_i["s"]
		del ep_i["s_"]
		_, samples = policy(cond, batch_size=1, verbose=False)
		obs_gen = samples.observations
		ep_i["s"] = obs_gen[0] # (T, obs_dim)
		ep_i["s_"] = np.concatenate([obs_gen[0][1:], obs_gen[0][-1:]], axis=0)
		res.append(ep_i)
	return res

def safefill_rollout(episodes_rollout):
	"""
	episodes_rollout:
		[{
			"s": (T, obs_dim),
			"act": (T, act_dim),
			"r": (T),
		}]]
	they could have different length, fill with the last frame to 
	make all have the same with the maximum length one
	for "s" and "act", repeat the last frame
	for "r", fill with 0
	"""
	max_len = max([len(ep["s"]) for ep in episodes_rollout])
	for i in range(len(episodes_rollout)):
		ep = episodes_rollout[i]
		for key in ["s", "act"]:
			if len(ep[key]) < max_len:
				ep[key] = np.concatenate([ep[key], np.repeat(ep[key][-1:], max_len - len(ep[key]), axis=0)], axis=0)
		if len(ep["r"]) < max_len:
			ep["r"] = np.concatenate([ep["r"], np.zeros(max_len - len(ep["r"]))], axis=0)
		episodes_rollout[i] = ep
	return episodes_rollout

"""others"""

OPEN_LARGE = \
		"############\\"+\
		"#OOOOOOOOOO#\\"+\
		"#OOOOOOOOOO#\\"+\
		"#OOOOOOOOOO#\\"+\
		"#OOOOOGOOOO#\\"+\
		"#OOOOOOOOOO#\\"+\
		"#OOOOOOOOOO#\\"+\
		"#OOOOOOOOOO#\\"+\
		"############"

register(
	id='maze2d-openlarge-v0',
	entry_point='d4rl.pointmaze:MazeEnv',
	max_episode_steps=800,
	kwargs={
		'maze_spec':OPEN_LARGE,
		'reward_type':'sparse',
		'reset_target': False,
		'ref_min_score': 6.7,
		'ref_max_score': 273.99,
		'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse-v1.hdf5'
	}
)

OPEN55 = \
		"#######\\"+\
		"#OOOOO#\\"+\
		"#OOOOO#\\"+\
		"#OOGOO#\\"+\
		"#OOOOO#\\"+\
		"#OOOOO#\\"+\
		"#######"

register(
	id='maze2d-open55-v0',
	entry_point='d4rl.pointmaze:MazeEnv',
	max_episode_steps=10000, # ! the origin value is 150
	kwargs={
		'maze_spec':OPEN55,
		'reward_type':'sparse',
		'reset_target': False,
		'ref_min_score': 0.01,
		'ref_max_score': 20.66,
		'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-open-sparse.hdf5'
	}
)

if __name__ == "__main__":
	pass
