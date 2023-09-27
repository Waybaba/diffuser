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
from src.modelmodule import eval_pair, rollout_ref
from diffuser.sampling.guides import *

"""functions"""

EVAL_START = [5.,2.,-0.,-0.]
# EVAL_START = [3.,3.,-0.,-0.]

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

class PlotMazeRunner:
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
		DEVICE = torch.device("cuda")

		### load modules
		print("Loading modules ...")
		diffuser = load_diffuser(cfg.diffuser.dir, cfg.diffuser.epoch)
		policy_func = cfg.policy
		guide_ = cfg.guide
		guide_ = NoTrainGuideXLower()
		diffusion, dataset, renderer= diffuser.net.diffusion, diffuser.dynamic_cfg["dataset"], diffuser.dynamic_cfg["dataset"].renderer
		policy = policy_func(
			guide=guide_,
			diffusion_model=diffusion,
			normalizer=dataset.normalizer,
		)
		model = policy.diffusion_model
		model.to(DEVICE)
		model.eval()

		### generate with condition
		if cfg.mode == "local":
			for scale in cfg.scale_list:
				for guide in cfg.guide_list:
					policy.guide = guide
					policy.sample_kwargs["scale"] = scale
					f_name = "maze"+ \
						f"-guide#{guide.__class__.__name__}" + \
						f"-scale#{str(scale)}"
					print("generating ", f_name)
					obs_list = self.generate(policy, cond={
						0: np.array(EVAL_START)
					}, batch_size=cfg.sample_num)
					obs_list = self.norm(obs_list, renderer.env_name)
					self.observations2fig(obs_list, Path(cfg.save_dir)/f"{f_name}.png", renderer)
					print(f"save to {Path(cfg.save_dir)/f'{f_name}.png'}")
		elif cfg.mode == "default":
			guide = cfg.guide
			scale = cfg.scale
			policy.guide = guide
			policy.sample_kwargs["scale"] = scale
			f_name = "maze"+ \
				f"-guide#{guide.__class__.__name__}" + \
				f"-scale#{str(scale)}"
			print("generating ", f_name)
			obs_list = self.generate(policy, cond={
				0: np.array([2.,5.,0.,0.])
			}, batch_size=cfg.sample_num)
			obs_list = self.norm(obs_list, renderer.env_name)
			img = self.observations2fig(obs_list, Path(cfg.save_dir)/f"{f_name}.png", renderer)
			self.observations2fig(obs_list, Path(cfg.output_dir)/f"{f_name}.png", renderer)
			print(f"save to {Path(cfg.save_dir)/f'{f_name}.png'}")
			# wandb.log({"maze": wandb.Image(img)}, commit=True)
			# change x,y direction
			wandb.log({"maze": wandb.Image(img.transpose(1,0,2))}, commit=True)
		

	def generate(self, policy, cond, batch_size):
		_, samples = policy(cond, batch_size=batch_size, verbose=False)
		obs_list = samples.observations[:batch_size]
		return obs_list

	def norm(self, obs_list, env_name):
		obs_list_normed = []
		from diffuser.utils.rendering import MAZE_BOUNDS
		bounds = MAZE_BOUNDS[env_name]
		for observations in obs_list:
			observations = observations + 0.5
			if len(bounds) == 2:
				_, scale = bounds
				observations /= scale
			elif len(bounds) == 4:
				_, iscale, _, jscale = bounds
				observations[:, 0] /= iscale
				observations[:, 1] /= jscale
			else:
				raise RuntimeError(f'Unrecognized bounds for {self.env_name}: {bounds}')
			obs_list_normed.append(observations)
		return obs_list_normed
	
	def observations2fig(self, obs_list, save_path, renderer):
		import matplotlib.pyplot as plt
		plt.clf()
		fig = plt.gcf()
		fig.set_size_inches(5, 5)
		plt.ioff()
		plt.imshow(renderer._background * 0.95,
			extent=renderer._extent, cmap=plt.cm.binary, vmin=0, vmax=1)
		# for each grid, plot a square which is slightly smaller than the grid
		# the total size is [-1, 1] for both x and y
		SIZE = 7
		GRID_LEN = 1 / SIZE
		GRID_FILL = 0.9
		for i in range(SIZE):
			for j in range(SIZE):
				# Calculate grid corner coordinates assuming total size is [-1, 1] for both x and y
				x_min = i * GRID_LEN + (1 - GRID_FILL) * GRID_LEN / 2
				y_min = j * GRID_LEN + (1 - GRID_FILL) * GRID_LEN / 2
				# Plot square that is slightly smaller than the grid (0.38x0.38)
				rect = plt.Rectangle((x_min, y_min), GRID_FILL * GRID_LEN, GRID_FILL * GRID_LEN, linewidth=0, edgecolor='none', facecolor='black', alpha=0.1)
				plt.gca().add_patch(rect)

		# random select 10
		NUM_LINE = 20
		for observations in np.array(obs_list)[np.random.choice(len(obs_list), NUM_LINE, replace=False)]:
			path_length = len(observations)
			colors = plt.cm.jet(np.linspace(1,0,path_length))
			# colors =sns.color_palette("husl", 8)
			# import seaborn as sns
			# colors = sns.color_palette("Paired", 12)
			plt.plot(observations[:,1], observations[:,0], c='white', zorder=10, alpha=1.0, lw=10.0) # bg
			# plt.plot(observations[:,1], observations[:,0], c='black', zorder=11, alpha=1.0, lw=5.0)
			# plt.plot(observations[:,1], observations[:,0], c='black', zorder=11, alpha=1.0, lw=5.0)
			plt.scatter(observations[:,1], observations[:,0], c=colors, zorder=20, s=60, alpha=1.0, edgecolors='none') # colorful
			# plot last point
			
		# plot start and end
		starts = [obs[0] for obs in obs_list]
		ends = [obs[-1] for obs in obs_list]
		plt.scatter(np.array(ends)[:,1], np.array(ends)[:,0], c='red', zorder=50, s=100, alpha=0.9, edgecolors='white', linewidths=2, marker="o") # end
		plt.scatter(np.array(starts)[:,1], np.array(starts)[:,0], c='green', zorder=50, s=300, edgecolors="white", linewidths=2, marker="*") # start
		# save
		MARGIN = 0.02
		plt.xlim(GRID_LEN-MARGIN, 1-GRID_LEN+MARGIN)
		plt.ylim(GRID_LEN-MARGIN, 1-GRID_LEN+MARGIN)
		plt.axis('off')
		# make dir for save_path parent if not exist
		save_dir = Path(save_path).parent
		save_dir.mkdir(parents=True, exist_ok=True)
		plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
		# plt.savefig(str(save_path).replace(".png", ".pdf"), bbox_inches='tight', pad_inches=0, dpi=300)
		# return rgb array
		from diffuser.utils.rendering import plot2img
		return plot2img(fig, remove_margins=False)


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
		wandb_logs["final/first_step_plan"] = wandb_media_wrapper(
			self.renderer.episodes2img(first_step_plan[:4], path=Path(cfg.output_dir)/"first_step_plan.png")
		)
		wandb_logs["final/rollout"] = wandb_media_wrapper(img_rollout_sample)
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