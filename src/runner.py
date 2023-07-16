
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
	def start(cfg):
		return
