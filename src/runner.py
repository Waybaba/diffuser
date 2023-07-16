import wandb

class TrainDiffuserRunner:
	
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

		### train

		n_epochs = int(cfg.global_cfg.n_train_steps // cfg.global_cfg.n_steps_per_epoch)
		for epoch in range(n_epochs):
			print(f'Epoch {epoch} / {n_epochs} | {cfg.output_dir}')
			trainer.train(n_train_steps=cfg.global_cfg.n_steps_per_epoch)
		
		print("Finished!")


class PlanGuidedRunner:
	def start(self, cfg):
		self.cfg = cfg
		import numpy as np
		
		diffusion_experiment = cfg.diffusion_experiment
		diffusion = diffusion_experiment.ema
		dataset = diffusion_experiment.dataset
		self.renderer = renderer = diffusion_experiment.renderer

		guide = cfg.guide

		policy = cfg.policy(
			guide=guide,
			diffusion_model=diffusion,
			normalizer=dataset.normalizer,
		)

		env = dataset.env
		observation = env.reset()

		## observations for rendering
		rollout = [observation.copy()]

		total_reward = 0
		for t in range(cfg.trainer.max_episode_length):

			if t % 10 == 0: print(cfg.output_dir, flush=True)

			## save state for rendering only
			state = env.state_vector().copy()

			## (optional) goal condition
			conditions = {0: observation}
			if "maze" in env.name:
				conditions[diffusion.horizon-1] = np.array(env.goal_locations[0] + env.goal_locations[0])
			action, samples = policy(conditions, batch_size=cfg.trainer.batch_size, verbose=cfg.trainer.verbose)

			## execute action in environment
			next_observation, reward, terminal, _ = env.step(action)

			## print reward and score
			total_reward += reward
			score = env.get_normalized_score(total_reward)
			print(
				f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
				f'values: {samples.values}',
				flush=True,
			)

			## update rollout observations
			rollout.append(next_observation.copy())

			## render every `cfg.trainer.vis_freq` steps
			self.log(t, samples, state, rollout, conditions)

			if terminal:
				break

			observation = next_observation
		return


	def log(self, t, samples, state, rollout=None, conditions=None):
		import os
		wandb_logs = {}

		if t % self.cfg.vis_freq != 0:
			return

		## render image of plans
		img_samples = self.renderer.composite(
			os.path.join(self.cfg.output_dir, f'{t}.png'),
			samples.observations,
			conditions
		)
		wandb_logs["samples"] = [wandb.Image(img_) for img_ in img_samples[0]]

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