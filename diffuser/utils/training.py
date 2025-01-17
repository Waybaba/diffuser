import os
import copy
import numpy as np
import torch
import einops
import pdb

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs
import wandb
import imageio

def cycle(dl):
	while True:
		for data in dl:
			yield data

class EMA():
	'''
		empirical moving average
	'''
	def __init__(self, beta):
		super().__init__()
		self.beta = beta

	def update_model_average(self, ma_model, current_model):
		for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
			old_weight, up_weight = ma_params.data, current_params.data
			ma_params.data = self.update_average(old_weight, up_weight)

	def update_average(self, old, new):
		if old is None:
			return new
		return old * self.beta + (1 - self.beta) * new

class Trainer(object):
	def __init__(
		self,
		diffusion_model,
		dataset,
		renderer,
		n_render_samples,
		ema_decay=0.995,
		train_batch_size=32,
		train_lr=2e-5,
		gradient_accumulate_every=2,
		step_start_ema=2000,
		update_ema_every=10,
		log_freq=100,
		sample_freq=1000,
		save_freq=1000,
		label_freq=100000,
		save_parallel=False,
		results_folder='./results',
		n_reference=8,
		bucket=None,
		task=None,
		condition_noise=0.0,
	):
		super().__init__()
		self.task = task
		self.model = diffusion_model
		self.ema = EMA(ema_decay)
		self.ema_model = copy.deepcopy(self.model)
		self.update_ema_every = update_ema_every

		self.step_start_ema = step_start_ema
		self.log_freq = log_freq
		self.sample_freq = sample_freq
		self.save_freq = save_freq
		self.label_freq = label_freq
		self.save_parallel = save_parallel
		self.n_render_samples = n_render_samples
		self.condition_noise = condition_noise

		self.batch_size = train_batch_size
		self.gradient_accumulate_every = gradient_accumulate_every

		self.dataset = dataset
		self.dataloader = cycle(torch.utils.data.DataLoader(
			self.dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=False
		))
		self.dataloader_vis = cycle(torch.utils.data.DataLoader(
			self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=False
		))
		self.renderer = renderer
		self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

		self.logdir = results_folder
		self.bucket = bucket
		self.n_reference = n_reference

		self.reset_parameters()
		self.step = 0

	def reset_parameters(self):
		self.ema_model.load_state_dict(self.model.state_dict())

	def step_ema(self):
		if self.step < self.step_start_ema:
			self.reset_parameters()
			return
		self.ema.update_model_average(self.ema_model, self.model)

	#-----------------------------------------------------------------------------#
	#------------------------------------ api ------------------------------------#
	#-----------------------------------------------------------------------------#

	def train(self, n_train_steps):

		timer = Timer()

		for step in range(n_train_steps):
			to_log = {}
			for i in range(self.gradient_accumulate_every):
				batch = next(self.dataloader)
				batch = batch_to_device(batch)
				### ! DEBUG apply noise to conditions
				# batch.conditions[0]: B,obs_dim
				# batch.conditions[1]: B,obs_dim
				# batch.trajectories: [B,T,act_dim+obs_dim]
				# applying shift to conditions and trajctories
				# set all actions to 0
				shift = torch.randn_like(batch.conditions[0]) * self.condition_noise
				for cond_k, cond_v in batch.conditions.items():
					if cond_k == 0: continue
					batch.conditions[cond_k] = batch.conditions[cond_k] + shift
				batch.trajectories[:,:,self.dataset.action_dim:] = batch.trajectories[:,:,self.dataset.action_dim:] + shift[:,None,:]
				batch.trajectories[:,:,:self.dataset.action_dim] = 0.0
				### !

				loss, infos = self.model.loss(*batch)
				loss = loss / self.gradient_accumulate_every
				loss.backward()

			self.optimizer.step()
			self.optimizer.zero_grad()

			### update ema
			if self.step % self.update_ema_every == 0:
				self.step_ema()

			### render reference
			if self.step == 0 and self.sample_freq:
				img_ref = self.render_reference(self.n_reference) # plot source data
				to_log["reference"] = [wandb.Image(img_ref)]
				
			### save model
			if self.step % self.save_freq == 0:
				assert self.task is not None, "task must be specified"
				label = self.step // self.label_freq * self.label_freq
				self.save(label)
			
			### log
			if self.step % self.log_freq == 0:
				infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
				to_log.update(infos)
				print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}', flush=True)

			### render samples (generate samples, rollout samples)
			if self.sample_freq and self.step % self.sample_freq == 0:
				assert self.task is not None, "task must be specified"
				img_samples, chain_samples = self.render_samples() # a [list of batch_size] with each one as one img but a composite one
				if chain_samples is not None: 
					to_log["chain"] = [wandb.Video(_) for _ in chain_samples]

				to_log["samples"] = [wandb.Image(img_) for img_ in img_samples]
				if self.task == "train_diffuser":
					total_reward_mean, img_rollout_samples = self.evals(self.model, num=5)
					to_log["eval/total_reward_mean"] = total_reward_mean
					to_log["eval/rollout"] = [wandb.Image(_) for _ in img_rollout_samples]

			### end
			self.step += 1
			wandb.log(to_log, step=self.step, commit=True if self.step % 1000 == 0 else False)

	def save(self, epoch):
		'''
			saves model and ema to disk;
			syncs to storage bucket if a bucket is specified
		'''
		data = {
			'step': self.step,
			'model': self.model.state_dict(),
			'ema': self.ema_model.state_dict()
		}
		savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
		torch.save(data, savepath)
		print(f'[ utils/training ] Saved model to {savepath}', flush=True)
		if self.bucket is not None:
			sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

	def load(self, epoch):
		'''
			loads model and ema from disk
		'''
		loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
		data = torch.load(loadpath)

		self.step = data['step']
		self.model.load_state_dict(data['model'])
		self.ema_model.load_state_dict(data['ema'])

	#-----------------------------------------------------------------------------#
	#--------------------------------- rendering ---------------------------------#
	#-----------------------------------------------------------------------------#

	def render_reference(self, batch_size=10):
		'''
			renders training points
		'''
		assert self.n_render_samples == 4, "please use 4, since we plot 4x1 for mujoco and 2x2 for maze"
		## get a temporary dataloader to load a single batch
		dataloader_tmp = cycle(torch.utils.data.DataLoader(
			self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=False
		))
		batch = dataloader_tmp.__next__()
		dataloader_tmp.close()

		## get trajectories and condition at t=0 from batch
		trajectories = to_np(batch.trajectories)
		conditions = to_np(batch.conditions[0])[:,None]

		## [ batch_size x horizon x observation_dim ]
		normed_observations = trajectories[:, :, self.dataset.action_dim:]
		observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

		savepath = os.path.join(self.logdir, f'_sample-reference.png')
		return self.renderer.composite(savepath, observations[:self.n_render_samples])

	def render_samples(self, batch_size=4):
		'''
			renders samples from (ema) diffusion model
			we sample batch_size conditions, 
			for each one in conditions,
			we generate n_samples trajectories with the same initial condition
			then plot them in a grid (2x2 for maze, 4x1 for mujoco)
			then we get batch_size images
			then would be save as:
				$UOURDIR/xxx/sample-{learning_step}-0.png
				$UOURDIR/xxx/sample-{learning_step}-1.png
				...
				$UOURDIR/xxx/sample-{learning_step}-{batch_size-1}.png
		'''
		assert self.n_render_samples == 4, "please use 4, since we plot 4x1 for mujoco and 2x2 for maze"
		n_samples = self.n_render_samples
		img_res = []
		chain_res = []
		for i in range(batch_size):

			## get a single datapoint
			batch = self.dataloader_vis.__next__()
			conditions = to_device(batch.conditions, 'cuda:0')

			### ! DEBUG apply noise to conditions
			# batch.conditions[0]: B,obs_dim
			# batch.conditions[1]: B,obs_dim
			# batch.trajectories: [B,T,act_dim+obs_dim]
			# applying shift to conditions and trajctories
			# set all actions to 0
			# shift = torch.randn_like(batch.conditions[0]) * self.condition_noise
			# for cond_k, cond_v in batch.conditions.items():
			#     if cond_k == 0: continue
			#     batch.conditions[cond_k] = batch.conditions[cond_k] + shift
			# batch.trajectories[:,:,self.dataset.action_dim:] = batch.trajectories[:,:,self.dataset.action_dim:] + shift[:,None,:]
			# batch.trajectories[:,:,:self.dataset.action_dim] = 0.0
			### !

			## repeat each item in conditions `n_samples` times
			conditions = apply_dict(
				einops.repeat,
				conditions,
				'b d -> (repeat b) d', repeat=n_samples,
			)

			## [ n_samples x horizon x (action_dim + observation_dim) ]
			samples = self.ema_model(conditions, return_chain=True)
			trajectories = to_np(samples.trajectories) # (n_samples, T, act_dim+obs_dim)
			chains = to_np(samples.chains) # (n_samples, horizon, T, act_dim+obs_dim)

			## [ n_samples x horizon x observation_dim ]
			normed_observations = trajectories[:, :, self.dataset.action_dim:] # (n_samples, T, obs_dim)
			normed_chains = chains[:, :, :, self.dataset.action_dim:] # (n_samples, horizon, T, obs_dim)

			# [ 1 x 1 x observation_dim ]
			normed_conditions = to_np(batch.conditions[0])[:,None]

			## [ n_samples x (horizon + 1) x observation_dim ]
			normed_observations = np.concatenate([
				np.repeat(normed_conditions, n_samples, axis=0),
				normed_observations
			], axis=1)

			## [ n_samples x (horizon + 1) x observation_dim ]
			observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
			chains = self.dataset.normalizer.unnormalize(normed_chains, 'observations')

			## render sample
			savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
			img_res.append(self.renderer.composite(savepath, observations))

			## render chains
			INTERVAL = 0.1
			chain_b = []
			if isinstance(INTERVAL, float): INTERVAL = int(INTERVAL*len(chains[0]))
			for t in range(0, len(chains[0]), INTERVAL):
				chain_b.append(self.renderer.composite(None, chains[:,t]))
			chain_b = np.stack(chain_b, axis=0) # (T_n, H, W, 3)
			chain_b = chain_b.transpose(0,3,1,2)
			chain_res.append(chain_b)
			savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.gif')
			T, Ch, H, W = chain_b.shape
			frames = []
			for t in range(T):
				frame = chain_b[t].transpose(1, 2, 0)
				frames.append(frame)
			imageio.mimsave(savepath, frames, 'GIF', duration=2.0/len(frames))
			print(f"Saved 4 samples to: {savepath}")


		return img_res, None if len(chain_res) == 0 else chain_res

	def eval(self, model, seed=3):
		env = self.dataset.env # goal and target are fixed
		diffusion = model
		env.seed(seed)
		observation = env.reset()
		if "maze" in env.name: env.set_target()
		## observations for rendering
		rollout = [observation.copy()]
		PLAN_ONCE = True
		BATCH_SIZE = 1
		USE_CONTROLLER_ACT = True
		USE_CONTROLLER_ACT = False if "maze" not in env.name else USE_CONTROLLER_ACT
		# fake policy
		from diffuser.sampling import GuidedPolicy, n_step_guided_p_sample
		from diffuser.sampling import NoTrainGuideShorter
		from functools import partial
		guide = NoTrainGuideShorter()
		policy = GuidedPolicy(
			guide=guide,
			diffusion_model=diffusion,
			normalizer=self.dataset.normalizer,
			scale=0.0, 
			preprocess_fns=[],
			sample_fn=partial(n_step_guided_p_sample,
				n_guide_steps=2, 
				t_stopgrad=2, 
				scale_grad_by_std=True, 
			),
		)

		total_reward = 0
		# for t in range(cfg.trainer.max_episode_length):
		for t in range(1000):
			wandb_logs = {}

			## save state for rendering only
			state = env.state_vector().copy()
			
			## make action
			conditions = {0: observation}
			if "maze" in env.name: 
				conditions[diffusion.horizon-1] = np.concatenate([env._target,[0, 0]])
			if t == 0: 
				actions, samples = policy(conditions, batch_size=BATCH_SIZE, verbose=False)
				action = samples.actions[0][0]
				sequence = samples.observations[0]
				first_conditions = conditions.copy()
			else:
				if not PLAN_ONCE:
					actions, samples = policy(conditions, batch_size=BATCH_SIZE, verbose=False)
					action = samples.actions[0][0]
					sequence = samples.observations[0]
			if USE_CONTROLLER_ACT:
				if t == diffusion.horizon - 1: 
					next_waypoint = sequence[-1].copy() if PLAN_ONCE else sequence[1]
					next_waypoint[2:] = 0
				else:
					next_waypoint = sequence[t+1] if PLAN_ONCE else sequence[1]
				action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
			
			## execute action in environment
			next_observation, reward, terminal, _ = env.step(action)
 
			## print reward and score
			total_reward += reward
			score = env.get_normalized_score(total_reward)
			
			## update rollout observations
			rollout.append(next_observation.copy())

			## render every `cfg.trainer.vis_freq` steps
			# self.log(t, samples, state, rollout, conditions)

			if terminal:
				break
			if PLAN_ONCE and t == diffusion.horizon - 1:
				break
			
			observation = next_observation
		
		### final log
		import os # TODO move
		wandb_logs = {}
		img_rollout_sample = self.renderer.render_rollout(
			os.path.join(self.logdir, f'rollout_final.png'),
			rollout,
			fps=80,
		)
		wandb_logs["final/rollout"] = wandb.Image(img_rollout_sample)
		wandb_logs["final/total_reward"] = total_reward
		guide_specific_metrics = guide.metrics(np.stack(rollout)[None,:])
		# for key, value in guide_specific_metrics.items():
		#     wandb_logs[f"final/guide_{key}"] = value[0].item()
		# wandb.log(wandb_logs)
		return total_reward, img_rollout_sample

	def evals(self, model, num=5):
		for seed in range(num):
			total_reward, img_rollout_sample = self.eval(model, seed=seed)
			if seed == 0:
				total_reward_list = [total_reward]
				img_rollout_samples = [img_rollout_sample]
			else:
				total_reward_list.append(total_reward)
				img_rollout_samples.append(img_rollout_sample)
		return np.mean(total_reward_list), img_rollout_samples
	
	def render_chains(self, chains, interval):
		return