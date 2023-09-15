import wandb

import hydra
from omegaconf import OmegaConf
from pathlib import Path
from copy import deepcopy
from collections import namedtuple
import numpy as np
import torch
from tqdm import tqdm
from diffuser.sampling import DummyGuide
from src.func import *
import numpy as np

"""functions"""
def rollout_ref(env, ep_ref, model, normalizer):
	""" rollout reference episodes
		TODO support different type of model, now it is
		env: the environment
		ep_ref: 
			1. {
				"s": (T, obs_dim),
				"s_": (T, obs_dim),
				"qpos": # optional for mujoco reset
				"qvel": # 
			} 
			2. (T, obs_dim)
				would be convert to 1.
			
		model: (obs_cur, obs_next) -> act
		for each step i, use current obs as obs_cur, use ep_ref[i] as obs_next
		act = model(obs_cur, obs_next)
		then return the rollout episodes with shape shape as ep_ref (T, obs_dim)
	! TODO there is error for last obs
	"""
	# convert if not dict
	if not isinstance(ep_ref, dict):
		raise ValueError("ep_ref should be a dict")
		ep_ref = {
			"s": np.stack(ep_ref),
			"s_": np.concatenate([ep_ref[1:], ep_ref[-1:]], axis=0),
		}
	# reset env with qpos, qvel
	if "qpos" in ep_ref:
		init_qpos = ep_ref["qpos"][0]
		init_qvel = ep_ref["qvel"][0]
		env.reset()
		env.set_state(init_qpos, init_qvel)
		s = env._get_obs()
		# env.sim.set_state(sim_state)
		# ss = env.state_vector()
		# # ss = env.reset()
		# print(ss)
		# s = ep_ref["s"][0]
		# s_ =  ep_ref["s_"][0]
		# print(s)
		# print(s_)
		# ! TODO have a check about this if equal
	else:
		s = env.reset()

	# run
	ep_s = []
	ep_a = []
	ep_r = []
	for env_i in tqdm(range(len(ep_ref["s"]))):
		device = next(model.parameters()).device
		model.to(device)
		if isinstance(model, FillActWrapper):
			act = model(torch.cat([
				torch.tensor(normalizer.normalize(
					s,
					"observations"
				)).to(device), 
				torch.tensor(normalizer.normalize(
					ep_ref["s_"][env_i],
					"observations"
				)).to(device)
			], dim=-1).float().to(device))
			act = act.detach().cpu().numpy()
		elif isinstance(model, EnvModelWrapper):
			act = model.act(
				torch.tensor(normalizer.normalize(
					s,
					"observations"
				)).to(device).float(),
				torch.tensor(normalizer.normalize(
					ep_ref["s_"][env_i],
					"observations"
				)).to(device).float()
			)
		act = normalizer.unnormalize(act, "actions")
		# act = ep_ref["act"][env_i] # ! DEBUG
		s_, r, done, info = env.step(act)
		ep_s.append(s)
		ep_a.append(act)
		ep_r.append(r)
		s = s_
		if done: break

	return {
		"s": np.stack(ep_s),
		"act": np.stack(ep_a),
		"r": np.stack(ep_r),
	}

def eval_pair(diffuser, controller=None, policy_func=None, plan_freq=None, guide_=None):
	### distill
	N_EPISODES = 1
	N_FULLROLLOUT = 1
	device = next(diffuser.net.parameters()).device
	diffusion, dataset, renderer= diffuser.net.diffusion, diffuser.dynamic_cfg["dataset"], diffuser.dynamic_cfg["dataset"].renderer
	policy = policy_func(
		guide=guide_,
		diffusion_model=diffusion,
		normalizer=dataset.normalizer,
	)
	policy_noguide = policy_func(
		guide=DummyGuide(),
		diffusion_model=diffusion, 
		normalizer=dataset.normalizer,
	)
	model = policy.diffusion_model
	model.to(device)
	model.eval()
	if controller is not None:
		actor = controller.net
		actor.to(device)
		actor.eval()
	env = dataset.env

	to_log = {}

	### episodes - generate
	episodes_ds = dataset.get_episodes_ref(num_episodes=N_EPISODES) # [{"s": ...}]
	episodes_diffuser = gen_with_same_cond(policy, episodes_ds) # [{"s": ...}]
	### episodes - rollout
	if controller is not None:
		episodes_ds_rollout = [rollout_ref(env, episodes_ds[i], actor, dataset.normalizer) for i in range(len(episodes_ds))]  # [{"s": ...}]
		episodes_diffuser_rollout = [rollout_ref(env, episodes_diffuser[i], actor, dataset.normalizer) for i in range(len(episodes_diffuser))]  # [{"s": ...}]
		episodes_full_rollout = [full_rollout_once(
			env, 
			policy, 
			actor, 
			dataset.normalizer, 
			plan_freq if isinstance(plan_freq, int) else max(int(plan_freq * model.horizon),1),
		) for i in range(N_FULLROLLOUT)]  # [{"s": ...}]
		episodes_ds_rollout = safefill_rollout(episodes_ds_rollout)
		episodes_diffuser_rollout = safefill_rollout(episodes_diffuser_rollout)
		episodes_full_rollout = safefill_rollout(episodes_full_rollout)

	### distill state
	states_ds = np.stack([each["s"] for each in episodes_ds], axis=0)
	states_diffuser = np.stack([each["s"] for each in episodes_diffuser], axis=0)
	if controller is not None:
		states_ds_rollout = np.stack([each["s"] for each in episodes_ds_rollout], axis=0)
		states_diffuser_rollout = np.stack([each["s"] for each in episodes_diffuser_rollout], axis=0)
		states_full_rollout = np.stack([each["s"] for each in episodes_full_rollout], axis=0)
	# unnormlize
	
	### cals common metric
	LOG_PREFIX = "value"
	LOG_SUB_PREFIX = "ds"
	metrics = guide_.metrics(states_ds)
	for k, v in metrics.items(): to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_{k}"] = v.mean()
	LOG_SUB_PREFIX = "diffuser"
	metrics = guide_.metrics(states_diffuser)
	for k, v in metrics.items(): to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_{k}"] = v.mean()
	if controller is not None:
		LOG_SUB_PREFIX = "ds_rollout"
		metrics = guide_.metrics(states_ds_rollout)
		for k, v in metrics.items(): to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_{k}"] = v.mean()
		LOG_SUB_PREFIX = "diffuser_rollout"
		metrics = guide_.metrics(states_diffuser_rollout)
		for k, v in metrics.items(): to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_{k}"] = v.mean()
		LOG_SUB_PREFIX = "full_rollout"
		metrics = guide_.metrics(states_full_rollout)
		for k, v in metrics.items(): to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_{k}"] = v.mean()

	### cals rollout metric
	if controller is not None:
		LOG_PREFIX = "value"
		LOG_SUB_PREFIX = "ds"
		r_sum = np.mean([each["r"].sum() for each in episodes_ds_rollout])
		to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_reward"] = r_sum
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
		renderer.episodes2img(states_ds[:4,:MAXSTEP])
	)]
	to_log[f"{LOG_PREFIX}/states_diffuser"] = [wandb.Image(
		renderer.episodes2img(states_diffuser[:4,:MAXSTEP])
	)]
	if controller is not None:
		to_log[f"{LOG_PREFIX}/states_ds_rollout"] = [wandb.Image(
			renderer.episodes2img(states_ds_rollout[:4,:MAXSTEP])
		)]
		to_log[f"{LOG_PREFIX}/states_diffuser_rollout"] = [wandb.Image(
			renderer.episodes2img(states_diffuser_rollout[:4,:MAXSTEP])
		)]
		to_log[f"{LOG_PREFIX}/states_full_rollout"] = [wandb.Image(
			renderer.episodes2img(states_full_rollout[:4,:MAXSTEP])
		)]
	return to_log

"""Runner"""
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
		diffuser = load_diffuser(cfg.diffuser.dir, cfg.diffuser.epoch)
		if cfg.controller.turn_on:
			controller = load_controller(cfg.controller.dir, cfg.controller.epoch)
			assert diffuser.dynamic_cfg["dataset"].env_name.split("-")[0] == controller.dynamic_cfg["dataset"].env_name.split("-")[0], \
				f"diffuser and controller should be trained on the same environment, while got {diffuser.dynamic_cfg['dataset'].env_name} and {controller.dynamic_cfg['dataset'].env_name}"
		print(f"\n\n\n### diffuser env loaded!: {diffuser.dynamic_cfg['dataset'].env_name}")
		# TODO controller could be null
		# TODO guide could be null
		if cfg.controller.turn_on:
			to_log = eval_pair(diffuser, controller, cfg.policy, cfg.plan_freq, cfg.guide)
		else:
			to_log = eval_pair(diffuser, None, cfg.policy, cfg.plan_freq, cfg.guide)
		wandb.log(to_log, commit=True)



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