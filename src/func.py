
"""Functions"""
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
		actor, 
		normalizer, 
		plan_freq=1,
		len_max=1000
	):
	"""
	planner: 
		call planner(cond, batch_size=1,verbose=False) and return actions, samples
	actor:
		call actor(obs, obs_, batch_size=1, verbose=False) and return act
	"""
		
	def make_act(actor, history, plan, t_madeplan, normalizer):
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

	def make_plan(planner, history):
		"""
		TODO: use history in guide
		"""
		cond = {
			0: history[-1]
		}
		actions, samples = planner(cond, batch_size=1,verbose=False)
		plan = samples.observations[0] # (T, obs_dim)
		return plan


	# assert actor.horizon >= plan_freq, "plan_freq should be smaller than horizon"
	assert actor.training == False, "actor should be in eval mode"
	assert planner.training == False, "planner should be in eval mode"
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
			plan = make_plan(planner, res["s"]+[s]) # (horizon, obs_dim)
			t_madeplan = env_step
		a = make_act(actor, res["s"]+[s], plan, t_madeplan, normalizer)
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
		"terminals": terminals
	}
	return env, dataset


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
