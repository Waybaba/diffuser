import wandb

from gym.envs.registration import register

import hydra
from omegaconf import OmegaConf
from pathlib import Path
from copy import deepcopy
from collections import namedtuple

import numpy as np
import torch
from .modelmodule import rollout_ref
from tqdm import tqdm
from diffuser.sampling import DummyGuide

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



class TrainDiffuserRunner:
	
	def start(self, cfg):
		print("Running default runner")
		self.cfg = cfg

		print("\n\n\n### loading datamodule ...")
		self.datamodule = cfg.datamodule()

		print("\n\n\n### loading modelmodule ...")
		self.modelmodule = cfg.modelmodule(
			dataset_info=self.datamodule.info,
		)

		print("\n\n\n### loading trainer ...")
		trainer = cfg.trainer(
			callbacks=[v for k,v in cfg.callbacks.items()],
			logger=[v for k,v in cfg.logger.items()],
		)
		
		print("\n\n\n### starting training ...")
		trainer.fit(
			model=self.modelmodule,
			datamodule=self.datamodule,
		)
		print("Finished!")

class TrainControllerRunner:
	
	def start(self, cfg):
		print("Running default runner")
		self.cfg = cfg

		self.datamodule = cfg.datamodule()
		self.modelmodule = cfg.modelmodule(
			dataset_info=self.datamodule.info,
		)
		trainer = cfg.trainer(
			callbacks=[v for k,v in cfg.callbacks.items()],
			logger=[v for k,v in cfg.logger.items()],
		)
		trainer.fit(
			model=self.modelmodule,
			datamodule=self.datamodule,
		)
		print("Finished!")

class TrainValuesRunner:
	
	def start(self, cfg):
		print("Running default runner")
		self.cfg = cfg

		### init	
		dataset = cfg.dataset()
		render = cfg.render()
		
		observation_dim = dataset.observation_dim
		action_dim = dataset.action_dim

		net = cfg.net(
			transition_dim=observation_dim + action_dim,
			cond_dim=observation_dim
		).to(cfg.device)
		
		model = cfg.model(
			net,
			observation_dim=observation_dim,
			action_dim=action_dim,
		).to(cfg.device)

		trainer = cfg.trainer(
			model,
			dataset,
			render,
		)
		
		# save diffuser training cfg for inference reload
		

		### train

		n_epochs = int(cfg.global_cfg.n_train_steps // cfg.global_cfg.n_steps_per_epoch)
		for epoch in range(n_epochs):
			print(f'Epoch {epoch} / {n_epochs} | {cfg.output_dir}')
			trainer.train(n_train_steps=cfg.global_cfg.n_steps_per_epoch)
		
		print("Finished!")

class EvalRunner:
	"""
	diffuser
	guide
	controller
	1. only generating 
	2. generating with guide
	3. control
	4. control with guide
	ps. everything is unnormlized in this class
	"""
	def start(self, cfg):
		self.cfg = cfg

		# load
		print("Loading modules ...")
		diffuser = self.load_diffuser(cfg.diffuser.dir, cfg.diffuser.epoch)
		controller = self.load_controller(cfg.controller.dir, cfg.controller.epoch)
		assert diffuser.dynamic_cfg["dataset"].env_name.split("-")[0] == controller.dynamic_cfg["dataset"].env_name.split("-")[0], \
			f"diffuser and controller should be trained on the same environment, while got {diffuser.dynamic_cfg['dataset'].env_name} and {controller.dynamic_cfg['dataset'].env_name}"
		print(f"\n\n\n### diffuser env loaded!: {diffuser.dynamic_cfg['dataset'].env_name}")
		# TODO controller could be null
		# TODO guide could be null

		### distill
		N_EPISODES = 4
		N_FULLROLLOUT = 4
		diffusion, dataset, self.renderer = diffuser.net.diffusion, diffuser.dynamic_cfg["dataset"], diffuser.dynamic_cfg["dataset"].renderer
		self.policy = cfg.policy(
			guide=cfg.guide,
			diffusion_model=diffusion,
			normalizer=dataset.normalizer,
		)
		self.policy_noguide = cfg.policy(
			guide=DummyGuide(),
			diffusion_model=diffusion, 
			normalizer=dataset.normalizer,
		)
		self.model = self.policy.diffusion_model
		self.model.to(cfg.device)
		self.model.eval()
		self.actor = controller.net
		self.actor.to(cfg.device)
		self.actor.eval()
		self.env = dataset.env

		to_log = {}

		### episodes - generate
		episodes_ds = dataset.get_episodes_ref(num_episodes=N_EPISODES) # [{"s": ...}]
		episodes_diffuser = self.gen_with_same_cond(episodes_ds) # [{"s": ...}]
		### episodes - rollout
		episodes_ds_rollout = [rollout_ref(self.env, episodes_ds[i], self.actor, dataset.normalizer) for i in range(len(episodes_ds))]  # [{"s": ...}]
		episodes_diffuser_rollout = [rollout_ref(self.env, episodes_diffuser[i], self.actor, dataset.normalizer) for i in range(len(episodes_diffuser))]  # [{"s": ...}]
		episodes_full_rollout = [self.full_rollout_once(
			self.env, 
			self.policy, 
			self.actor, 
			dataset.normalizer, 
			cfg.plan_freq if isinstance(cfg.plan_freq, int) else max(int(cfg.plan_freq * self.model.horizon),1),
		) for i in range(N_FULLROLLOUT)]  # [{"s": ...}]

		### distill state
		states_ds = np.stack([each["s"] for each in episodes_ds], axis=0)
		states_diffuser = np.stack([each["s"] for each in episodes_diffuser], axis=0)
		states_ds_rollout = np.stack([each["s"] for each in episodes_ds_rollout], axis=0)
		states_diffuser_rollout = np.stack([each["s"] for each in episodes_diffuser_rollout], axis=0)
		states_full_rollout = np.stack([each["s"] for each in episodes_full_rollout], axis=0)
		# unnormlize
		
		### cals common metric
		LOG_PREFIX = "value"
		LOG_SUB_PREFIX = "ds"
		metrics = cfg.guide.metrics(states_ds)
		for k, v in metrics.items(): to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_{k}"] = v.mean()
		LOG_SUB_PREFIX = "diffuser"
		metrics = cfg.guide.metrics(states_diffuser)
		for k, v in metrics.items(): to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_{k}"] = v.mean()
		LOG_SUB_PREFIX = "ds_rollout"
		metrics = cfg.guide.metrics(states_ds_rollout)
		for k, v in metrics.items(): to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_{k}"] = v.mean()
		LOG_SUB_PREFIX = "diffuser_rollout"
		metrics = cfg.guide.metrics(states_diffuser_rollout)
		for k, v in metrics.items(): to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_{k}"] = v.mean()
		LOG_SUB_PREFIX = "full_rollout"
		metrics = cfg.guide.metrics(states_full_rollout)
		for k, v in metrics.items(): to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_{k}"] = v.mean()

		### cals rollout metric
		LOG_PREFIX = "value"
		LOG_SUB_PREFIX = "ds_rollout"
		r_sum = np.mean([each["r"].sum() for each in episodes_ds_rollout])
		to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_reward"] = r_sum
		LOG_SUB_PREFIX = "diffuser_rollout"
		r_sum = np.mean([each["r"].sum() for each in episodes_diffuser_rollout])
		to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_reward"] = r_sum
		LOG_SUB_PREFIX = "full_rollout"
		r_sum = np.mean([each["r"].sum() for each in episodes_full_rollout])
		to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_reward"] = r_sum


		### render
		LOG_PREFIX = "val_ep_end"
		# STEPS = min(len(episodes_rollout[0]["s"]), len(episodes_ds_rollout[0]["s"]), 32)
		MAXSTEP = 200
		to_log[f"{LOG_PREFIX}/states_ds"] = [wandb.Image(
			self.renderer.episodes2img(states_ds[:4,:MAXSTEP])
		)]
		to_log[f"{LOG_PREFIX}/states_diffuser"] = [wandb.Image(
			self.renderer.episodes2img(states_diffuser[:4,:MAXSTEP])
		)]
		to_log[f"{LOG_PREFIX}/states_ds_rollout"] = [wandb.Image(
			self.renderer.episodes2img(states_ds_rollout[:4,:MAXSTEP])
		)]
		to_log[f"{LOG_PREFIX}/states_diffuser_rollout"] = [wandb.Image(
			self.renderer.episodes2img(states_diffuser_rollout[:4,:MAXSTEP])
		)]
		to_log[f"{LOG_PREFIX}/states_full_rollout"] = [wandb.Image(
			self.renderer.episodes2img(states_full_rollout[:4,:MAXSTEP])
		)]
		wandb.log(to_log, commit=True)
		
	def generate(self, conditions={}, repeat=1):
		"""
		return:
			Trajectories(actions, observations, samples.values)
		"""
		actions, samples = self.policy(conditions, batch_size=repeat)
		return samples.observations

	def gen_with_same_cond(self, episodes_ds):
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
			obs_gen = self.generate(cond, repeat=1) # samples (B, T, obs_dim)
			ep_i["s"] = obs_gen[0] # (T, obs_dim)
			ep_i["s_"] = np.concatenate([obs_gen[0][1:], obs_gen[0][-1:]], axis=0)
			res.append(ep_i)
		return res
	
	def load_diffuser(self, dir_, epoch_):
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

	def load_controller(self, dir_, epoch_):
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
			self, 
			env, 
			planner, 
			actor, 
			normalizer, 
			plan_freq=1,
			len_max=1000
		):
		"""
			env: 
			time: 
		"""
		assert self.model.horizon >= plan_freq, "plan_freq should be smaller than horizon"
		print(f"Start full rollout, plan_freq={plan_freq}, len_max={len_max} ...")
		res = {
			"act": [],
			"s": [],
			"s_": [],
			"r": [],
		}
		env_step = 0

		t_madeplan = -99999
		
		s = env.reset()
		while True: 
			if env_step - t_madeplan >= plan_freq:
				plan = self.make_plan(planner, res["s"]+[s]) # (horizon, obs_dim)
				t_madeplan = env_step
			a = self.make_act(actor, res["s"]+[s], plan, t_madeplan, normalizer)
			s_, r, done, info = env.step(a)
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
			
	def make_act(self, actor, history, plan, t_madeplan, normalizer):
		"""
		actor: would generate act, different for diff methods
		history: [obs_dim]*t_cur # note the length should be t_cur so that plan would be made
		"""
		s = history[-1]
		s_ = plan[len(history)-1-t_madeplan] # e.g. for first step, len(history)=1, t_madeplan=0, we should use first element of plan as s_
		model = actor
		device = next(actor.parameters()).device
		model.to(device)
		act = model(torch.cat([
			torch.tensor(normalizer.normalize(
				s,
				"observations"
			)).to(device), 
			torch.tensor(normalizer.normalize(
				s_,
				"observations"
			)).to(device)
		], dim=-1).float().to(device))
		act = act.detach().cpu().numpy()
		act = normalizer.unnormalize(act, "actions")
		return act

	def make_plan(self, planner, history):
		"""
		TODO: use history in guide
		"""
		cond = {
			0: history[-1]
		}
		actions, samples = planner(cond, batch_size=1,verbose=False)
		plan = samples.observations[0] # (T, obs_dim)
		return plan



def parse_diffusion(diffusion_dir, epoch, device, dataset_seed):
	""" parse diffusion model from 
	"""
	diffusion_hydra_cfg_path = diffusion_dir + "/hydra_config.yaml"
	with open(diffusion_hydra_cfg_path, "r") as file:
		cfg = OmegaConf.load(file)
	cfg = hydra.utils.instantiate(cfg)
	### init	
	dataset = cfg.dataset(seed=dataset_seed)
	render = cfg.render(dataset.env.name)
	
	observation_dim = dataset.observation_dim
	action_dim = dataset.action_dim

	net = cfg.net(
		transition_dim=observation_dim + action_dim,
		cond_dim=observation_dim
	).to(device)
	
	model = cfg.model(
		net,
		observation_dim=observation_dim,
		action_dim=action_dim,
	).to(device)

	cfg.trainer.keywords['results_folder'] = diffusion_dir
	trainer = cfg.trainer(
		model,
		dataset,
		render,
	)
	# find latest
	import glob
	def get_latest_epoch(loadpath):
		states = glob.glob1(loadpath, 'state_*')
		latest_epoch = -1
		for state in states:
			epoch = int(state.replace('state_', '').replace('.pt', ''))
			latest_epoch = max(epoch, latest_epoch)
		return latest_epoch
	if epoch == 'latest':
		epoch = get_latest_epoch(diffusion_dir)
	print(f'\n[ parse diffusion model ] Loading model epoch: {epoch}\n')
	print(f'\n[ parse diffusion model ] Path: {diffusion_dir}\n')
	trainer.load(epoch)

	return trainer.ema_model, dataset, render

import numpy as np

class PlanGuidedRunner:
	CUSTOM_TARGET = {
		"tl2br": {
			"location": (1.0, 1.0),
			"target": np.array([7, 10]),
		},
		"br2tl": {
			"location": (7.0, 10.0),
			"target": np.array([1, 1]),
		},
		"tr2bl": {
			"location": (7.0, 1.0),
			"target": np.array([1, 10]),
		},
		"bl2tr": {
			"location": (1.0, 10.0),
			"target": np.array([7, 1]),
		},
		"2wayAv1": {
			"location": (1.0, 1.0),
			"target": np.array([3, 4]),
		},
		"2wayAv2": {
			"location": (3.0, 4.0),
			"target": np.array([1, 1]),
		},
		"2wayBv1": {
			"location": (5, 6),
			"target": np.array([1, 10]),
		},
		"2wayBv2": {
			"location": (1, 10.0),
			"target": np.array([5, 6]),
		},
	}
	def start(self, cfg):
		self.cfg = cfg

		diffuser = self.load_diffuser(cfg.diffuser.dir, cfg.diffuser.epoch)
		diffusion, dataset, self.renderer = diffuser.net.diffusion, diffuser.dynamic_cfg["dataset"], diffuser.dynamic_cfg["dataset"].renderer
		
		# diffusion, dataset, self.renderer = parse_diffusion(cfg.diffusion.dir, cfg.diffusion.epoch, cfg.device, cfg.diffusion.dataset_seed)

		guide = cfg.guide

		policy = cfg.policy(
			guide=guide,
			diffusion_model=diffusion,
			normalizer=dataset.normalizer,
		)

		policy.diffusion_model.to(cfg.device)

		env = dataset.env
		if "maze" not in env.name:
			assert cfg.trainer.custom_target is None, "Only maze environments need targets, so the cfg.trainer.custom_target should be None"
			assert cfg.trainer.use_controller_act is False, "Only maze environments can use controller, so the cfg.trainer.use_controller_act should be False"
		
		if cfg.trainer.custom_target is not None:
			# env.set_state(self.CUSTOM_TARGET[cfg.trainer.custom_target]["state"])
			env.set_target(self.CUSTOM_TARGET[cfg.trainer.custom_target]["target"])
			observation = env.reset_to_location(self.CUSTOM_TARGET[cfg.trainer.custom_target]["location"])
			print("###")
			print(f"use custom target ### {cfg.trainer.custom_target} ###")
			print("state", env.state_vector())
			print("target", env._target)
			print()
		else:
			observation = env.reset()

		## observations for rendering
		rollout = [observation.copy()]
		
		total_reward = 0
		for t in range(cfg.trainer.max_episode_length):
			wandb_logs = {}
			if t % 10 == 0: print(cfg.output_dir, flush=True)

			## save state for rendering only
			state = env.state_vector().copy()
			
			## make action
			conditions = {0: observation}
			if "mazexxx" in env.name: 
				conditions[diffusion.horizon-1] = np.array(list(env._target) + [0, 0])
			if t == 0: 
				actions, samples = policy(conditions, batch_size=cfg.trainer.batch_size, verbose=cfg.trainer.verbose)
				act_env = samples.actions[0][0] # (a_dim) only select one for act_
				sequence = samples.observations[0] # (horizon, s_dim)
				first_step_plan = samples.observations # (B, horizon, s_dim)
				first_step_conditions = conditions

				# import matplotlib.pyplot as plt
				# import torch
				# path="./debug/trace.png"
				# plt.figure()
				# x_ = samples.observations[:,:,[6,14,15]]
				# turn to cpu numpy
				# if isinstance(x_, torch.Tensor):
				#     if x_.is_cuda:
				#         x_ = x_.cpu()
				#     x_ = x_.detach().numpy()
				# coord_0 = x_[0]
				# T = coord_0.shape[0]
				# for dim in range(coord_0.shape[1]):
				# 	plt.plot(np.arange(T), coord_0[:, dim])
				# plt.savefig(path)
			else:
				if not cfg.trainer.plan_once:
					actions, samples = policy(conditions, batch_size=cfg.trainer.batch_size, verbose=cfg.trainer.verbose)
					act_env = samples.actions[0][0]
					sequence = samples.observations[0]
			if cfg.trainer.use_controller_act:
				if t == diffusion.horizon - 1: 
					next_waypoint = sequence[-1].copy() if cfg.trainer.plan_once else sequence[1]
					next_waypoint[2:] = 0
				else:
					next_waypoint = sequence[t+1] if cfg.trainer.plan_once else sequence[1]
				act_env = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
			
			## execute action in environment
			next_observation, reward, terminal, _ = env.step(act_env)
 
			## print reward and score
			total_reward += reward
			score = env.get_normalized_score(total_reward)
			print(
				f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
				f'values: {samples.values[0].item()}',
				flush=True,
			)
			wandb_logs["rollout/reward"] = reward
			wandb_logs["rollout/total_reward"] = total_reward
			wandb_logs["rollout/values"] = score
			guide_specific_metrics = guide.metrics(samples.observations)
			for key, value in guide_specific_metrics.items():
				wandb_logs[f"rollout/guide_{key}"] = value[0].item()

			## update rollout observations
			rollout.append(next_observation.copy())

			## render every `cfg.trainer.vis_freq` steps
			# self.log(t, samples, state, rollout, first_step_conditions)

			## wandb log
			wandb_commit = (not cfg.wandb.lazy_commits_freq) or (t % cfg.wandb.lazy_commits_freq == 0)
			wandb.log(wandb_logs, step=t, commit=wandb_commit)

			## end
			if terminal:
				break
			if cfg.trainer.plan_once and t == diffusion.horizon - 1:
				break
			
			observation = next_observation
		
		### final log
		import os # TODO move
		wandb_logs = {}
		img_rollout_sample = self.renderer.render_rollout(
			os.path.join(self.cfg.output_dir, f'rollout_final.png'),
			rollout,
			first_step_conditions,
			fps=80,
		)
		guide_specific_metrics = guide.metrics(first_step_plan)
		for key, value in guide_specific_metrics.items():
			wandb_logs[f"final/first_step_plan_{key}"] = value[0].item()
		wandb_logs["final/first_step_plan"] = wandb.Image(
			self.renderer.episodes2img(first_step_plan[:4], path=Path(cfg.output_dir)/"first_step_plan.png")
		)
		wandb_logs["final/rollout"] = wandb.Image(img_rollout_sample)
		wandb_logs["final/total_reward"] = total_reward
		guide_specific_metrics = guide.metrics(np.stack(rollout)[None,:])
		for key, value in guide_specific_metrics.items():
			wandb_logs[f"final/guide_{key}"] = value[0].item()
		wandb.log(wandb_logs)

	def log(self, t, samples, state, rollout=None, conditions=None):
		import os
		wandb_logs = {}

		if t % self.cfg.vis_freq != 0:
			return

		## render image of plans
		img_sample = self.renderer.composite(
			os.path.join(self.cfg.output_dir, f'{t}.png'),
			samples.observations[:4],
			conditions
		)
		# wandb_logs["samples"] = [wandb.Image(img_) for img_ in img_samples[0]]
		wandb_logs["samples"] = [wandb.Image(img_sample)]

		# render video of plans
		# self.renderer.render_plan(
		#     os.path.join(self.cfg.output_dir, f'{t}_plan.mp4'),
		#     samples.actions[:self.cfg.max_render],
		#     samples.observations[:self.cfg.max_render],
		#     state,
		#     conditions, 
		# )

		if rollout is not None:
			## render video of rollout thus far
			img_sample = self.renderer.render_rollout(
				os.path.join(self.cfg.output_dir, f'rollout.png'),
				rollout,
				conditions,
				fps=80,
			)
			wandb_logs["rollout"] = wandb.Image(img_sample)
		
		wandb.log(wandb_logs)

	def finish(self, t, score, total_reward, terminal, diffusion_experiment, value_experiment):
		import os
		import json
		json_path = os.path.join(self.cfg.output_dir, 'rollout.json')
		json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
			'epoch_diffusion': diffusion_experiment.epoch, 'epoch_value': value_experiment.epoch}
		json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
		print(f'[ utils/logger ] Saved log to {json_path}')
	
	def load_diffuser(self, dir_, epoch_):
		print("\n\n\n### loading diffuser ...")
		from src.modelmodule import DiffuserModule
		diffuser_cfg = OmegaConf.load(Path(dir_)/"hydra_config.yaml")
		datamodule = hydra.utils.instantiate(diffuser_cfg.datamodule)()
		modelmodule = DiffuserModule.load_from_checkpoint(
			Path(dir_)/"checkpoints"/f"{epoch_}.ckpt",
			dataset_info=datamodule.info,
		)
		return modelmodule