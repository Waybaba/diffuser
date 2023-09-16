
		if False:
			### full rollout
			self.actor = self.controller.net
			self.actor.to(self.device)
			self.actor.eval()
			self.policy = GuidedPolicy(
				guide=DummyGuide(),
				diffusion_model=self.net.diffusion, 
				normalizer=dataset.normalizer,
				preprocess_fns=[],
				scale=0.0,
				n_guide_steps=1, # ! does not used, only use one step + time travel
				sample_fn=n_step_guided_p_sample_freedom_timetravel,
				t_stopgrad=2, # positive: grad[t < t_stopgrad] = 0; bigger is noise
				scale_grad_by_std=True,
				travel_repeat=1, # time travel
				travel_interval=[0.0,1.0], # if float, would use [horizon*travel_interval, horizon]
			)
			self.policy.diffusion_model.eval()
			episodes_full_rollout = [full_rollout_once(
				env, 
				self.policy, 
				self.actor, 
				dataset.normalizer, 
				# self.hparams.plan_freq if isinstance(self.hparams.plan_freq, int) else max(int(self.hparams.plan_freq * dataset.kwargs["horizon"]),1),
				self.hparams.plan_freq if isinstance(self.hparams.plan_freq, int) else max(int(self.hparams.plan_freq * dataset.kwargs["horizon"]),1),
			) for i in range(N_FULLROLLOUT)]  # [{"s": ...}]

			### ds ref rollout
			N_EPISODES = 1
			episodes_ds = dataset.get_episodes_ref(num_episodes=N_EPISODES) # [{"s": ...}]
			episodes_diffuser = gen_with_same_cond(self.policy, episodes_ds) # [{"s": ...}]
			episodes_ds_rollout = [rollout_ref(self.env, episodes_ds[i], self.actor, dataset.normalizer) for i in range(len(episodes_ds))]  # [{"s": ...}]
			episodes_diffuser_rollout = [rollout_ref(self.env, episodes_diffuser[i], self.actor, dataset.normalizer) for i in range(len(episodes_diffuser))]  # [{"s": ...}]
			
			episodes_ds_rollout = safefill_rollout(episodes_ds_rollout)
			episodes_diffuser_rollout = safefill_rollout(episodes_diffuser)
			episodes_full_rollout = safefill_rollout(episodes_full_rollout)
			
			### cals rollout metric
			LOG_PREFIX = "value"
			LOG_SUB_PREFIX = "full_rollout"
			MAXSTEP = 200
			r_sum = np.mean([each["r"].sum() for each in episodes_full_rollout])
			to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_reward"] = r_sum
			states_full_rollout = np.stack([each["s"] for each in episodes_full_rollout], axis=0)
			to_log[f"{LOG_PREFIX}/states_full_rollout"] = [wandb.Image(
				self.dynamic_cfg["dataset"].renderer.episodes2img(states_full_rollout[:4,:MAXSTEP])
			)]
