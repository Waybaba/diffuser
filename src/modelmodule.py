from typing import Any, List
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MinMetric, Metric
import torch.nn as nn
from torchmetrics.classification.accuracy import Accuracy
import torch
from tqdm import tqdm
import numpy as np
import wandb
import inspect
import einops
from diffuser.utils.arrays import batch_to_device, to_np, to_device, apply_dict
from src.datamodule import EpisodeBatch, EpisodeValidBatch
import random
from src.func import *
from diffuser.sampling.guides import DummyGuide
from diffuser.sampling.policies import GuidedPolicy
from diffuser.sampling import n_step_guided_p_sample_freedom_timetravel, n_step_guided_p_sample
# from src.runner import eval_pair, rollout_ref
import functools


"""functions"""


def collect_parameters(model, set="all"):
	""" Collect parameters from model depending on set.
	Args: 
		model: model to collect parameters from
		set: "all" or "backbone"
			e.g.
			"all": all parameters
			"backbone": only backbone parameters
			"head": only head parameters
			"decoder": only decoder parameters
			"encoder": only encoder parameters
			"bn": only batch normalization parameters
			"default": depends on the model
		lr: learning rate for the parameters # TODO
	"""
	# assert is wrapper (even when custom model is used)
	assert isinstance(model, ModelWrapperBase)
	# collect parameters
	if set == "all": params = model.parameters()
	elif set in ["backbone", "head", "decoder", "decode_head", "special"]:
		params = model.select_param_group(set)
	elif set in ["decoder","decode_head"]: 
		params = model.decode_head.parameters()
	elif set == "bn": 
		params = []
		for name, p in model.named_parameters():
			if "bn" in name: params.append(p)
		if len(params) == 0: raise ValueError("No batch normalization parameters found")
	elif set == "default": params = model.parameters()
	elif set == "10xforhead":
		# 10x learning rate for classification head, 1x for other layers in decoder
		# para_head_names = ["conv_seg.weight", "conv_seg.bias"]
		# classification_head = list(filter(lambda kv: kv[0] in para_head_names, model.decode_head.named_parameters()))
		# other_decoder_layers = list(filter(lambda kv: kv[0] not in para_head_names, model.decode_head.named_parameters()))
		# classification_head = [i[1] for i in classification_head]
		# other_decoder_layers = [i[1] for i in other_decoder_layers]
		# params = [{"params": classification_head, "lr": lr * 10}, {"params": other_decoder_layers, "lr": lr}]
		raise NotImplementedError("10xforhead is deprecated, use two sets instead")
	elif set in ["generator", "discriminator"]:
		params = model.select_param_group(set)
	else: raise ValueError(f"set {set} is not supported")
	# set requires_grad
	# for p in model.parameters(): p.requires_grad = False
	# params = [i for i in params]
	# for p in params:
	# 	if isinstance(p, dict): 
	# 		for p_ in p["params"]: p_.requires_grad = True
	# 	else: p.requires_grad = True
	return params

class L1DistanceMetric(Metric):
	"""Metric to calculate L1 distance."""
	
	def __init__(self):
		super().__init__()
		# Initialize the state variables to store the sum and the count of samples.
		self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
		self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")
		
	def update(self, preds: torch.Tensor, target: torch.Tensor):
		"""
			Receives the output of the model and the target.
			return mean l1 distance of a dimmension
		"""
		assert preds.shape == target.shape, "preds and target should have the same shape"
		# assert len(preds.shape) == 2, "preds and target should be 2D"
		with torch.no_grad():
			l1_distance = torch.abs(preds - target).sum()
			self.sum += l1_distance
			self.count += target.numel()
		return l1_distance / target.numel()
	
	def compute(self):
		"""Compute the metric."""
		return self.sum / self.count

class Controller:
	""" Include a series of controlle which can be init then give actions
	mode:
		str which indicate the controller type
		1. random: random action
		2. ss2a###{controller_run_dir}:
			predict P(a|s,s')
		3. mpc###{contoller_run_dir}
			use a P(s'|s,a) model and MPC algorithm to act
	act:
		would take all kinds of input for all usages
	"""
	def __init__(self, mode, *args, **kwargs):
		self.args, self.kwargs = args, kwargs
		self.mode = mode
		if self.mode == "random":
			assert "act_dim" in kwargs, "act_dim should be in kwargs"
		else:
			raise ValueError(f"mode {mode} not supported")

	def act(self, *args, **kwargs):
		if self.mode == "random":
			return torch.random.randint(0, self.kwargs["act_dim"], size=(1,))
		else:
			raise ValueError(f"mode {self.mode} not supported")

def eval_pair(diffuser, controller=None, policy_func=None, plan_freq=None, guide_=None):
	### distill
	N_EPISODES = 1
	N_FULLROLLOUT = 1
	device = torch.device("cuda")
	diffusion, dataset, renderer= diffuser.net.diffusion, diffuser.dynamic_cfg["dataset"], diffuser.dynamic_cfg["dataset"].renderer
	policy = policy_func(
		guide=guide_,
		diffusion_model=diffusion,
		normalizer=dataset.normalizer,
	)
	normalizer_actor = controller.dynamic_cfg["dataset"].normalizer
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
		episodes_ds_rollout = [rollout_ref(env, episodes_ds[i], actor, normalizer_actor) for i in range(len(episodes_ds))]  # [{"s": ...}]
		episodes_diffuser_rollout = [rollout_ref(env, episodes_diffuser[i], actor, normalizer_actor) for i in range(len(episodes_diffuser))]  # [{"s": ...}]
		episodes_full_rollout = [full_rollout_once(
			env, 
			policy, 
			actor, 
			normalizer_actor, 
			plan_freq if isinstance(plan_freq, int) else max(int(plan_freq * model.horizon)-1,1),
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
	MAXSTEP = 1000
	to_log[f"{LOG_PREFIX}/states_ds"] = [wandb_media_wrapper(
		renderer.episodes2img(states_ds[:4,:MAXSTEP])
	)]
	to_log[f"{LOG_PREFIX}/states_diffuser"] = [wandb_media_wrapper(
		renderer.episodes2img(states_diffuser[:4,:MAXSTEP])
	)]
	if controller is not None:
		to_log[f"{LOG_PREFIX}/states_ds_rollout"] = [wandb_media_wrapper(
			renderer.episodes2img(states_ds_rollout[:4,:MAXSTEP])
		)]
		to_log[f"{LOG_PREFIX}/states_diffuser_rollout"] = [wandb_media_wrapper(
			renderer.episodes2img(states_diffuser_rollout[:4,:MAXSTEP])
		)]
		to_log[f"{LOG_PREFIX}/states_full_rollout"] = [wandb_media_wrapper(
			renderer.episodes2img(states_full_rollout[:4,:MAXSTEP])
		)]
	return to_log

def eval_diffuser_witha(diffuser, policy_func=None, plan_freq=None, guide_=None):
	### distill
	N_EPISODES = 1
	N_FULLROLLOUT = 1
	device = torch.device("cuda")
	diffusion, dataset, renderer = diffuser.net.diffusion, diffuser.dynamic_cfg["dataset"], diffuser.dynamic_cfg["dataset"].renderer
	policy = policy_func(
		guide=guide_,
		diffusion_model=diffusion,
		normalizer=dataset.normalizer,
	)
	# normalizer_actor = controller.dynamic_cfg["dataset"].normalizer
	model = policy.diffusion_model
	model.to(device)
	model.eval()

	env = dataset.env

	to_log = {}

	### episodes - generate
	episodes_ds = dataset.get_episodes_ref(num_episodes=N_EPISODES) # [{"s": ...}]
	episodes_diffuser = gen_with_same_cond(policy, episodes_ds) # [{"s": ...}]
	### episodes - rollout
	episodes_full_rollout = [full_rollout_once_witha(
		env, 
		policy, 
		dataset.normalizer, 
		plan_freq if isinstance(plan_freq, int) else max(int(plan_freq * model.horizon)-1,1),
	) for i in range(N_FULLROLLOUT)]  # [{"s": ...}]
	episodes_full_rollout = safefill_rollout(episodes_full_rollout)

	### distill state
	states_full_rollout = np.stack([each["s"] for each in episodes_full_rollout], axis=0)
	states_ds = np.stack([each["s"] for each in episodes_ds], axis=0)
	# unnormlize
	### cals common metric
	LOG_PREFIX = "value"
	LOG_SUB_PREFIX = "full_rollout"
	metrics = guide_.metrics(states_full_rollout)
	for k, v in metrics.items(): to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_{k}"] = v.mean()

	### cals rollout metric
	LOG_PREFIX = "value"
	LOG_SUB_PREFIX = "ds"
	to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_reward"] = np.mean([each["r"].sum() for each in episodes_ds])
	to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_length"] = states_ds.shape[1]
	LOG_SUB_PREFIX = "full_rollout"
	to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_reward"] = np.mean([each["r"].sum() for each in episodes_full_rollout])
	to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_length"] = states_full_rollout.shape[1]

	### render
	LOG_PREFIX = "val_ep_end"
	MAXSTEP = 1000
	to_log[f"{LOG_PREFIX}/states_full_rollout"] = [wandb_media_wrapper(
		renderer.episodes2img(states_full_rollout[:4,:MAXSTEP])
	)]
	return to_log

def eval_bc(diffuser):
	### distill
	N_EPISODES = 1
	N_FULLROLLOUT = 1
	device = torch.device("cuda")
	bc_model, dataset, renderer = diffuser.net, diffuser.dynamic_cfg["dataset"], diffuser.dynamic_cfg["dataset"].renderer
	# normalizer_actor = controller.dynamic_cfg["dataset"].normalizer
	model = bc_model
	model.to(device)
	model.eval()

	env = dataset.env

	to_log = {}

	### episodes - generate
	episodes_ds = dataset.get_episodes_ref(num_episodes=N_EPISODES) # [{"s": ...}]
	### episodes - rollout
	episodes_full_rollout = [full_rollout_once_bc(
		env, 
		model, 
		dataset.normalizer, 
	) for i in range(N_FULLROLLOUT)]  # [{"s": ...}]
	episodes_full_rollout = safefill_rollout(episodes_full_rollout)

	### distill state
	states_full_rollout = np.stack([each["s"] for each in episodes_full_rollout], axis=0)
	states_ds = np.stack([each["s"] for each in episodes_ds], axis=0)
	# unnormlize
	### cals common metric
	LOG_PREFIX = "value"
	LOG_SUB_PREFIX = "full_rollout"
	# for k, v in metrics.items(): to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_{k}"] = v.mean()

	### cals rollout metric
	LOG_PREFIX = "value"
	LOG_SUB_PREFIX = "ds"
	to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_reward"] = np.mean([each["r"].sum() for each in episodes_ds])
	to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_length"] = states_ds.shape[1]
	LOG_SUB_PREFIX = "full_rollout"
	to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_reward"] = np.mean([each["r"].sum() for each in episodes_full_rollout])
	to_log[f"{LOG_PREFIX}/{LOG_SUB_PREFIX}_length"] = states_full_rollout.shape[1]

	### render
	LOG_PREFIX = "val_ep_end"
	MAXSTEP = 1000
	to_log[f"{LOG_PREFIX}/states_full_rollout"] = [wandb_media_wrapper(
		renderer.episodes2img(states_full_rollout[:4,:MAXSTEP])
	)]
	return to_log


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
		if isinstance(s, tuple): s = s[0]

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
		tmp = env.step(act)
		# s_, r, done, info 
		if len(tmp) == 4: s_, r, done, info = tmp
		else:
			s_, r, ter, timeout, info = tmp
			done = ter or timeout
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

"""model wrapper"""
class ModelWrapperBase(nn.Module):
	""" 
	###
	TLDR
		A common api for all models from different frameworks.
	###
	Why?
		There are many available models online while they can not be directly applied. 
		The application process is depending on the framework it comes from. Here, we 
		provide a common api for all models from different frameworks.
		1. replacing the classifier head to adapt the class num
		2. get trainable parameters
	Details
		The root class is ModelWrapperBase. It is a wrapper for all models from different frameworks.
		Then, for each task (e.g. sl_seg, sl_cls, ttda_seg), we have a parent wrapper class.
		Then, for each framework (e.g. timm, torch), we have a child wrapper class.
		The struncture is like this:
			ModelWrapperBase
			|--- SLClsModelWrapper
			|	|--- TimmSLClsModelWrapper
			|	|--- TorchSLClsModelWrapper
			|--- SLSegModelWrapper
			|	|--- MmsegSLSegModelWrapper
			|	|--- TorchSLSegModelWrapper
			|--- TTDASegModelWrapper
			...
		SLClsModelWrapper:
			1. use dynamic_cfg["cls_num"] to replace the classifier head
			2. select_param_group("backbone") support "backbone" and "head"
		SLSegModelWrapper:
			... (see SLClsModelWrapper)
	ps. classes for framework and wrapper for method such as token prompt should be considered separately
	
	__init__:
		model_class: a partial function which returns a model
			should be instantiated with dynamic_cfg as model_class(dynamic_cfg)
		dynamic_cfg: a dictionary with dynamic configuration
			dynamic_cfg would be recursively passed to multiple wrappers
			dynamic_cfg is passed during the initialization of the Model, which
			can includes some dynamic configuration, e.g. number of classes
		**kwargs: other arguments for this wrapper, such as bias=True,False for 
			the classifier head, or n_ctx for the visual prompt
	"""
	def __init__(self, model_class=None, dynamic_cfg=None, **kwargs):
		super().__init__()
		# if model_class is a partial function, and model_class.func is a class, and the class is subclass ofModelWrapperBase 
		if model_class is None: # pass in None means no submodel
			pass
		elif isinstance(model_class, functools.partial): # partial function
			print(f"init_model: go to next layer: {model_class}")
			if inspect.isclass(model_class.func):
				if issubclass(model_class.func, ModelWrapperBase):
					self.model = model_class(dynamic_cfg=dynamic_cfg) # Wrapper
				else:
					self.model = model_class()  # class e.g. timm.models.vision_transformer.VisionTransformer
			elif inspect.isfunction(model_class.func): # function e.g. timm.create_model
				self.model = model_class()
			else:
				raise ValueError(f"model_class should be a partial function, but got {model_class}")
		else:
			raise ValueError(f"model_class should be a partial function, but got {model_class}")
	
	def forward(self, x):
		raise NotImplementedError

	def select_param_group(self, name):
		raise NotImplementedError

	def torch_module_init(self):
		super().__init__()

class S2AWrapper(ModelWrapperBase):
	"""
	Abstract class, need to set self.in_dim, self.outdim in later class
	"""
	def __init__(self, dynamic_cfg, **kwargs):
		assert hasattr(self, "in_dim") and hasattr(self, "out_dim"), "in_dim and out_dim should be set in later class"
		self.torch_module_init()
		self.net = kwargs["net"]
		# if partial, give input dim
		if isinstance(self.net[0], functools.partial):
			self.net[0] = self.net[0](in_features=self.in_dim)
		if isinstance(self.net[-1], functools.partial):
			self.net[-1] = self.net[-1](out_features=self.out_dim)
		if kwargs["tahn"]:
			self.net.append(nn.Tanh())
		self.net = torch.nn.Sequential(*self.net)
	
	def forward(self, x):
		if len(x.shape) == 1: 
			x = x.unsqueeze(0)
			return self.net(x).squeeze(0)
		return self.net(x)

	def select_param_group(self, name):
		raise NotImplementedError

class FillActWrapper(S2AWrapper):
	""" s -> a
	"""
	def __init__(self, dynamic_cfg, **kwargs):
		self.in_dim = dynamic_cfg["obs_dim"] * 2
		self.out_dim = dynamic_cfg["act_dim"]
		super().__init__(dynamic_cfg, **kwargs)

class BCNetWrapper(S2AWrapper):
	""" (s,s) -> a
	"""
	def __init__(self, dynamic_cfg, **kwargs):
		self.in_dim = dynamic_cfg["obs_dim"]
		self.out_dim = dynamic_cfg["act_dim"]
		super().__init__(dynamic_cfg, **kwargs)

class EnvModelWrapper(ModelWrapperBase):
	def __init__(self, dynamic_cfg, **kwargs):
		self.torch_module_init()
		self.dynamic_cfg = dynamic_cfg
		self.act_dim = dynamic_cfg["act_dim"]
		self.obs_dim = dynamic_cfg["obs_dim"]
		in_dim = dynamic_cfg["obs_dim"] + dynamic_cfg["act_dim"]
		out_dim = dynamic_cfg["obs_dim"]
		self.net = kwargs["net"]
		self.net[0] = self.net[0](in_features=in_dim)
		self.net[-1] = self.net[-1](out_features=out_dim)
		self.net = torch.nn.Sequential(*self.net)
	
	def forward(self, x):
		if len(x.shape) == 1: 
			x = x.unsqueeze(0)
			return self.net(x).squeeze(0)
		return self.net(x)

	def select_param_group(self, name):
		raise NotImplementedError

	def act(self, x):
		return x
	
class DiffusionWrapper(ModelWrapperBase):
	def __init__(self, dynamic_cfg, **kwargs):
		self.torch_module_init()
		self.net = kwargs["net"](
			transition_dim=dynamic_cfg["obs_dim"] + dynamic_cfg["act_dim"],
			cond_dim=dynamic_cfg["obs_dim"],
		)
		self.diffusion = kwargs["diffusion"](
			model=self.net,
			observation_dim=dynamic_cfg["obs_dim"],
			action_dim=dynamic_cfg["act_dim"],
		)
	
	def forward(self, *args, **kwargs):
		return self.diffusion(*args, **kwargs)
	
	def loss(self, *args, **kwargs):
		return self.diffusion.loss(*args, **kwargs)

	def select_param_group(self, name):
		raise NotImplementedError

"""plightning modelmodule"""

class DefaultModule(LightningModule):
	"""Example of LightningModule for MNIST classification.

	A LightningModule organizes your PyTorch code into 6 sections:
		- Computations (init).
		- Train loop (training_step)
		- Validation loop (validation_step)
		- Test loop (test_step)
		- Prediction Loop (predict_step)
		- Optimizers and LR Schedulers (configure_optimizers)

	Read the docs:
		https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
	"""

	def __init__(
		self,
		**kwargs
	):
		super().__init__()

		# this line allows to access init params with 'self.hparams' attribute
		# also ensures init params will be stored in ckpt
		self.save_hyperparameters(logger=False, ignore=["dataset_info"])

		# setup dynamic config
		self.dynamic_cfg = dynamic_cfg = self.init_dynamic_cfg(kwargs["dataset_info"])
		self.net = self.hparams.net(dynamic_cfg = dynamic_cfg)
		assert isinstance(self.net, ModelWrapperBase), "net should always be a ModelWrapperBase, check src.models.default_model.ModelWrapperBase"

		# use separate metric instance for train, val and test step
		# to ensure a proper reduction over the epoch
		self.train_acc = self.hparams.metric_func().cpu()
		self.val_acc = nn.ModuleList([self.hparams.metric_func().cpu() for _ in dynamic_cfg["data_val"]])
		self.test_acc = nn.ModuleList([self.hparams.metric_func().cpu() for _ in dynamic_cfg["data_test"]])
		
		# for logging best so far validation accuracy
		self.val_acc_best = nn.ModuleList([MinMetric()for _ in dynamic_cfg["data_val"]])
		self.val_acc_best_mean = MinMetric()

		# log dataset length
		len_log = {}
		if "data_train" in dynamic_cfg:
			self.log("once/len_train", len(dynamic_cfg["data_train"]))
		if "data_val" in dynamic_cfg:
			self.log("once/len_val", sum([len(each) for each in dynamic_cfg["data_val"]]))
		if "data_test" in dynamic_cfg:
			self.log("once/len_test", sum([len(each) for each in dynamic_cfg["data_test"]]))
		if "dataset" in dynamic_cfg:
			self.log("once/len_dataset", len(dynamic_cfg["dataset"]))
		if len_log: wandb.log(len_log)

	def on_train_start(self):
		# by default lightning executes validation step sanity checks before training starts,
		# so we need to make sure val_acc_best doesn't store accuracy from these checks
		self.val_acc_best_mean.reset()
		for val_acc_best in self.val_acc_best: val_acc_best.reset()

	def step(self, batch: Any):
		""" process the batch from dataloader and return the res_batch
		input: `batch dict` from dataloader
		output: `res_batch` dict with "x", "y", "info", "outputs", "preds", "loss"
		ps. note that the calcultion of "outputs", "preds", "loss" could
			be task-specific, so we need to implement it in the subclass
		"""
		raise NotImplementedError

	def training_step(self, batch: Any, batch_idx: int):
		batch = self.step(batch)

		# log train metrics
		acc = self.train_acc(batch["preds"], batch["y"])
		self.log("train/loss_step", batch["loss"], on_step=True, on_epoch=False, prog_bar=True)
		self.log("train/acc_step", acc, on_step=True, on_epoch=False, prog_bar=True)

		# we can return here dict with any tensors
		# and then read it in some callback or in `training_epoch_end()` below
		# remember to always return loss from `training_step()` or else backpropagation will fail!
		return {"loss": batch["loss"]}

	def training_epoch_end(self, outputs: List[Any]):
		# `outputs` is a list of dicts returned from `training_step()`
		train_acc = self.train_acc.compute()
		self.log("train/acc/epoch_compute", train_acc, prog_bar=True)
		self.train_acc.reset()
	
	def on_val_dataloader_start(self, dataloader_idx: int):
		self.val_acc[dataloader_idx].reset()

	def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
		batch = self.step(batch)
		if batch_idx == 0: 
			self.on_val_dataloader_start(dataloader_idx)
		# log val metrics
		acc = self.val_acc[dataloader_idx](batch["preds"], batch["y"])
		self.log("val/loss_step", batch["loss"], on_step=True, on_epoch=False, prog_bar=False, add_dataloader_idx=True)
		self.log("val/acc_step", acc, on_step=True, on_epoch=False, prog_bar=True, add_dataloader_idx=True)

		return {"loss": batch["loss"]}

	def validation_epoch_end(self, outputs: List[Any]):
		"""
		here, we only hand `mean` and `best`, others are handled in validation_step
		1. (None) epoch acc is already handled by validation_step
		2. calculate `best for now acc` for each in val_list
		3. calculate `mean acc epoch` for each in val_list 
		4. calculate `best for now mean acc epoch` for each in val_list
		"""
		val_accs = [val_acc.compute() for val_acc in self.val_acc]
		# best (seperately)
		for i in range(len(val_accs)): # log accs of each dataset in val list
			self.val_acc_best[i].update(val_accs[i])
			self.log(f"val/acc/dataloaderr_compute_idx_{str(i)}", val_accs[i], on_epoch=True, prog_bar=True)
			self.log(f"val/acc/dataloaderr_compute_idx_{str(i)}_best", self.val_acc_best[i].compute(), on_epoch=True, prog_bar=True)
		
		# mean and best (mean)
		acc_mean = sum(val_accs) / len(val_accs)
		self.log("val/acc/mean", acc_mean, on_step=False, on_epoch=True, prog_bar=True) # log mean
		self.val_acc_best_mean.update(acc_mean)
		self.log("val/acc/mean_best", self.val_acc_best_mean.compute(), on_epoch=True, prog_bar=True) # log best mean
		
		# for model checkpoint (use first one)
		self.log("val/acc", val_accs[0], on_step=False, on_epoch=True, prog_bar=False) 

		# reset
		for i, val_acc in enumerate(self.val_acc):
			val_acc.reset()

	def on_test_dataloader_start(self, dataloader_idx: int):
		self.test_acc[dataloader_idx].reset()

	def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
		if batch_idx == 0: 
			self.on_test_dataloader_start(dataloader_idx)
		batch = self.step(batch)
		# log test metrics
		acc = self.test_acc[dataloader_idx](batch["preds"], batch["y"])
		self.log(f"test/loss_step", batch["loss"], on_step=True, on_epoch=False, prog_bar=True, add_dataloader_idx=True)
		self.log(f"test/acc_step", acc, on_step=True, on_epoch=False, prog_bar=True, add_dataloader_idx=True)

		return {"loss": batch["loss"]}

	def test_epoch_end(self, outputs: List[Any]):
		test_accs = [test_acc.compute() for test_acc in self.test_acc]
		acc_mean = sum(test_accs) / len(test_accs)
		self.log("test/acc/mean", acc_mean, on_step=False, on_epoch=True, prog_bar=True)
		for i in range(len(test_accs)): # log accs of each dataset in val list
			self.log(f"test/acc/dataloader_compute_idx_{str(i)}", test_accs[i], on_epoch=True, prog_bar=True)
		for i, test_acc in enumerate(self.test_acc): # reset
			test_acc.reset()

	def configure_optimizers(self):
		"""Choose what optimizers and learning-rate schedulers to use in your optimization.
		Input:
			self.hparams.optimizations: list of dict, each dict contains:
				param_target: str, "all", "decoder", ...
				optimizer: torch.optim.Optimizer
				lr_scheduler_config: dict, contains:
					scheduler: a partial function of torch.optim.lr_scheduler, waiting for optimizer as input
					interval: str, "step" or "epoch"
					frequency: int, how many steps/epochs between two scheduler updates
		Output:
			two lists, one for optimizers, one for lr_schedulers
		ps. for different lr for different layers, use multiple optimizers, each would
			be applied on a subset of parameters
		"""
		optimizers = []
		lr_schedulers = []
		for optimization in self.hparams.optimizations:
			if optimization["param_target"] is None: continue
			target_params =collect_parameters(
						model=self.net, 
						set=optimization["param_target"]
			)
			optimizer = optimization["optimizer"](target_params)
			lr_scheduler = optimization["lr_scheduler_config"] if "lr_scheduler_config" in optimization else None
			if lr_scheduler is not None:
				lr_scheduler["scheduler"] = lr_scheduler["scheduler"](optimizer)
				lr_scheduler = {k: v for k, v in lr_scheduler.items()}
			else:
				lr_scheduler = None
			optimizers.append(optimizer)
			lr_schedulers.append(lr_scheduler)
		# freeze non-trainable parameters for saving memory
		# self.net.requires_grad_(False)
		# for op in optimizers:
		# 	for p in op.param_groups[0]["params"]:
		# 		p.requires_grad = True
		assert len([s is None for s in lr_schedulers]) in [0, len(lr_schedulers)], "lr_schedulers should be all None or all not None"
		if lr_schedulers[0] is None: return optimizers
		return optimizers, lr_schedulers

	def init_dynamic_cfg(self, ds_info):
		"""init dynamic cfg
		task specific.
		e.g. for classification, we need to init the number of classes
		for segmentation, we need to init the number of classes and class names
		"""
		return ds_info

	@property
	def wandb(self):
		for lg in self.loggers:
			if "wandb" in lg.__module__:
				return lg
		raise ValueError("No wandb logger found")

	def torch_module_init(self):
		super().__init__()

class SLModule(DefaultModule):
	pass

class SLClassificationModule(DefaultModule):
	def step(self, batch: Any):
		"""
		input: `batch dict` from dataloader
		output: `res_batch` dict with "x", "y", "info", "outputs", "preds", "loss"
		ps. note that the calcultion of "outputs", "preds", "loss" could
			be task-specific, so we need to implement it in the subclass
		"""
		# extract data from batch
		res_batch = {}
		if isinstance(batch, list) or isinstance(batch, tuple): # for normal datasets
			if len(batch) == 3:
				res_batch["x"], res_batch["y"], res_batch["info"] = batch
			else:
				res_batch["x"], res_batch["y"] = batch
				res_batch["info"] = None
		elif isinstance(batch, dict): # for MMSegWrapper
			if isinstance(batch["img"], list): # for test of multiple size images
				res_batch["x"], res_batch["y"] = batch["img"][0], batch["gt_semantic_seg"][0]
				res_batch["info"] = {k: v[0] for k, v in batch["img_metas"].items()}
			else: 
				res_batch["x"], res_batch["y"], res_batch["info"] = batch["img"], batch["gt_semantic_seg"], batch["img_metas"]
		else:
			raise ValueError("Unknown batch type")
		# ["outputs"]
		res_batch["outputs"], info = self.net(res_batch["x"])
		# ["preds"]
		logits = res_batch["outputs"]
		res_batch["preds"] = torch.argmax(logits, dim=1)
		# ["loss"]
		logits = res_batch["outputs"]
		y = res_batch["y"]
		res_batch["loss"] = self.hparams.loss_func()(logits, y)
		return res_batch

	def init_dynamic_cfg(self, ds_info):
		"""
		The return should be:
		dynamic_cfg:
			classnames:
			n_cls:
		"""
		dynamic_cfg = {}
		if not hasattr(ds_info["data_train"], "info"):
			raise ValueError(
				"""
				The info() function is not implemented in the dataset class
				"""
			)
		dynamic_cfg.update(ds_info["data_train"].info())
		return dynamic_cfg

class FillActModelModule(DefaultModule):
	def step(self, batch: Any):
		""" process the batch from dataloader and return the res_batch
		input: `batch dict` from dataloader
		output: `res_batch` dict with "x", "y", "info", "outputs", "preds", "loss"
		ps. note that the calcultion of "outputs", "preds", "loss" could
			be task-specific, so we need to implement it in the subclass
		"""
		# s, s_, act
		outputs = self.net(torch.cat([batch.s, batch.s_], dim=-1))
		res_batch = {
			"s": batch.s,
			"s_": batch.s_,
			"act": batch.act,
			# "info": batch["info"],
			"outputs": outputs,
			"preds": outputs,
			"loss": self.hparams.loss_func()(outputs, batch.act),
			"y": batch.act,
		}
		return res_batch

	def validation_epoch_end(self, outputs):
		assert self.net.training == False, "net should be in eval mode"
		LOG_PREFIX = "val_ep_end"
		super().validation_epoch_end(outputs)
		
		### rollout -> [(T, obs_dim)]
		episodes_ref = self.dynamic_cfg["dataset"].get_episodes_ref(num_episodes=1)
		episodes_rollout = [rollout_ref(self.dynamic_cfg["env"], ep_ref, self.net, self.dynamic_cfg["dataset"].normalizer) for ep_ref in episodes_ref]
		episodes_rollout = safefill_rollout(episodes_rollout)

		### cals metric
		metrics = self.cal_ref_rollout_metrics(episodes_ref, episodes_rollout)
		for k, v in metrics.items(): self.log(f"{LOG_PREFIX}/{k}", v, on_epoch=True, prog_bar=True)
		
		### render
		# STEPS = min(80, *[len(ep) for ep in episodes_ref])
		MAXSTEP = 1000
		states_ref = np.stack([each["s"] for each in episodes_ref], axis=0)
		states_rollout = np.stack([each["s"] for each in episodes_rollout], axis=0)
		to_log = {}
		to_log[f"{LOG_PREFIX}/ref"] = [wandb_media_wrapper(
			self.dynamic_cfg["renderer"].episodes2img(states_ref[:4,:MAXSTEP])
		)]
		to_log[f"{LOG_PREFIX}/rollout"] = [wandb_media_wrapper(
			self.dynamic_cfg["renderer"].episodes2img(states_rollout[:4,:MAXSTEP])
		)]
		wandb.log(to_log)
	
	def cal_ref_rollout_metrics(self, episodes_ref, episodes_rollout):
		""" cal ref rollout metrics
			episodes_ref: a list of reference episodes [(T_i, obs_dim)]
			episodes_rollout: a list of rollout episodes [(T_i, obs_dim)]
			return a dict of metrics
		"""
		to_log = {
			# "mean_l1_shift_total": np.mean([
			# 	L1DistanceMetric()(torch.tensor(episodes_ref[i]["s"]), torch.tensor(episodes_rollout[i]["s"])) \
			# 		for i in range(len(episodes_ref))
			# ]),
			# "mean_l1_shift_20steps": np.mean([
			# 	L1DistanceMetric()(torch.tensor(episodes_ref[i]["s"][:20]), torch.tensor(episodes_rollout[i]["s"][:20])) \
			# 		for i in range(len(episodes_ref))
			# ]),
			# "mean_l1_shift_80steps": np.mean([
			# 	L1DistanceMetric()(torch.tensor(episodes_ref[i]["s"][:80]), torch.tensor(episodes_rollout[i]["s"][:80])) \
			# 		for i in range(len(episodes_ref))
			# ]),
			"sum_reward_total": np.mean([
				np.sum(episodes_rollout[i]["r"]) \
				for i in range(len(episodes_ref))
			]),
			"sum_reward_20steps": np.mean([
				np.sum(episodes_rollout[i]["r"][:20]) \
				for i in range(len(episodes_ref))
			]),
			"sum_reward_80steps": np.mean([
				np.sum(episodes_rollout[i]["r"][:80]) \
				for i in range(len(episodes_ref))
			]),
			"len_rollout": episodes_rollout[0]["r"].shape[0],
			"len_ds": episodes_ref[0]["r"].shape[0],
		}
		if "r" in episodes_ref[0]:
			to_log["sum_reward_total_ds"] = np.mean([
				np.sum(episodes_ref[i]["r"]) \
				for i in range(len(episodes_ref))
			])
		return to_log

	def render_composite(self, states, renderer, path=None,steps=40):
		"""
		steps controls the number of steps in the animation
			int: the first steps frames are rendered
			float: (<1) select the intervals of i*steps*len(states)
		states: (B, T, obs_dim)
		"""
		# TODO controller, the lenght could be different
		if isinstance(steps, int): steps = np.arange(steps)
		elif isinstance(steps, float): steps = np.arange(int(len(states) * steps))
		else: raise ValueError(f"steps should be int or float, but got {steps}")
		img = renderer.composite(path, states[:, steps])
		return img

	def act(self, s, s_):
		"""
		assume s, s_ are unnormalized
		"""

class EnvModelModule(FillActModelModule):
	def step(self, batch: Any):
		""" process the batch from dataloader and return the res_batch
		input: `batch dict` from dataloader
		output: `res_batch` dict with "x", "y", "info", "outputs", "preds", "loss"
		ps. note that the calcultion of "outputs", "preds", "loss" could
			be task-specific, so we need to implement it in the subclass
		"""
		# s, s_, act
		outputs = self.net(torch.cat([batch.s, batch.act], dim=-1))
		res_batch = {
			"s": batch.s,
			"s_": batch.s_,
			"act": batch.act,
			# "info": batch["info"],
			"outputs": outputs,
			"preds": outputs,
			"loss": self.hparams.loss_func()(outputs, batch.s_),
			"y": batch.s_,
		}
		return res_batch

	def validation_epoch_end(self, outputs):
		assert self.net.training == False, "net should be in eval mode"
		LOG_PREFIX = "val_ep_end"
		super().validation_epoch_end(outputs)
		### render a plot
		# get ref episode from dataset (T, obs_dim)
		episodes_ref = self.get_ref_episodes(self.dynamic_cfg["env"], ep_num=10)
		# rollout to get [(T, obs_dim)]
		raise ValueError("need to implement rollout for this")
		episodes_rollout = [rollout_ref(self.dynamic_cfg["env"], ep_ref, self.net, self.dynamic_cfg["dataset"].normalizer) for ep_ref in episodes_ref]
		# metric
		metrics = self.cal_ref_rollout_metrics(episodes_ref, episodes_rollout)
		for k, v in metrics.items():
			self.log(f"{LOG_PREFIX}/{k}", v, on_epoch=True, prog_bar=True)
		# render
		states_ref = np.stack([each["s"] for each in episodes_ref], axis=0)
		states_rollout = np.stack([each["s"] for each in episodes_rollout], axis=0)
		self.wandb.log_image(f"{LOG_PREFIX}/ref", [wandb.Image(
			self.render_composite(states_ref[:4], self.dynamic_cfg["renderer"](),steps=80)
		)])
		self.wandb.log_image(f"{LOG_PREFIX}/rollout", [wandb.Image(
			self.render_composite(states_rollout[:4], self.dynamic_cfg["renderer"](),steps=80)
		)])

# Diffuser
class DiffuserModule(DefaultModule):
	def __init__(
		self,
		**kwargs
	):
		super().__init__(**kwargs)
		if "controller" in self.hparams and self.hparams.controller.turn_on:
			self.controller = load_controller(self.hparams.controller.dir, self.hparams.controller.epoch)
		
	def step(self, batch: Any):
		""" process the batch from dataloader and return the res_batch
		input: `batch dict` from dataloader
		output: `res_batch` dict with "x", "y", "info", "outputs", "preds", "loss"
		ps. note that the calcultion of "outputs", "preds", "loss" could
			be task-specific, so we need to implement it in the subclass
		"""
		# s, s_, act
		kwargs = {}
		if len(batch) == 3: kwargs["valids"] = batch.valids
		loss, _ = self.net.loss(batch.trajectories, batch.conditions, **kwargs)
		# applying nosie robustness
		if self.hparams.data_noise: 
			batch = batch._replace(trajectories=batch.trajectories + torch.randn_like(batch.trajectories) * self.hparams.data_noise)
			for k, v in batch.conditions.items():
				batch.conditions[k] += torch.randn_like(v) * self.hparams.data_noise
		res_batch = {
			"outputs": batch.trajectories, # TODO
			"preds": batch.trajectories, # TODO
			"y": batch.trajectories, # TODO
			"loss": loss
		}
		return res_batch

	def validation_epoch_end(self, outputs):
		### init
		assert self.net.training == False, "net should be in eval mode"
		LOG_PREFIX = "val_ep_end"
		to_log = {}

		### get render data # TODO spilt well
		dataset = self.dynamic_cfg["dataset"]
		env = dataset.env
		ref_samples, img_samples, chain_samples = self.render_samples() # a [list of batch_size] with each one as one img but a composite one
		# ! TODO get full ep
		N_FULLROLLOUT = 1
		# if self.hparams.eval.turn_on:
		if self.hparams.controller.turn_on:
			print("Start eval_pair")
			to_log_ = eval_pair(self, self.controller, self.hparams.controller.policy, plan_freq=self.hparams.controller.plan_freq, guide_=self.hparams.controller.guide)
			to_log.update({"eval_pair/"+k: v for k, v in to_log_.items()})
		### log
		LOG_PREFIX="val_ep_end"
		to_log["ref"] = [wandb_media_wrapper(_) for _ in ref_samples]
		if chain_samples is not None: 
			to_log["chain"] = [wandb_media_wrapper(_) for _ in chain_samples]
		to_log["samples"] = [wandb_media_wrapper(_) for _ in img_samples]
		wandb.log({
			f"{LOG_PREFIX}/{k}": v for k, v in to_log.items()
		}, commit=True)
		super().validation_epoch_end(outputs)

	def render_samples(self):
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
		batch_size = 1
		N_SAMPLES = 4 # would have same condition, rendered in one img
		ref_res, img_res, chain_res = [], [], []
		dataset = self.dynamic_cfg["data_val"][0]
		from torch.utils.data.dataloader import default_collate
		from collections.abc import Mapping, Sequence

		def recursive_collate(batch):
			elem = batch[0]
			if isinstance(elem, Mapping):
				return {key: recursive_collate([d[key] for d in batch]) for key in elem}
			elif isinstance(elem, Sequence) and not isinstance(elem, str):
				return [recursive_collate(samples) for samples in zip(*batch)]
			else:
				return default_collate(batch)  # use PyTorch's default_collate

		for i in range(batch_size):
			
			## get a single datapoint
			batch = [dataset[random.randint(0, len(dataset)-1)]]
			batch = recursive_collate(batch) # stack another dim
			batch = EpisodeValidBatch(*batch) if len(batch) == 3 else EpisodeBatch(*batch)
			conditions = batch.conditions
			refs = to_np(batch.trajectories)

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
				'b ... -> (repeat b) ...', repeat=N_SAMPLES,
			)
			refs = einops.repeat(refs, 'b ... -> (repeat b) ...', repeat=N_SAMPLES)


			## [ n_samples x horizon x (action_dim + observation_dim) ]
			samples = self.net(conditions, return_chain=True) # ! ADD EMA in paper
			trajectories = to_np(samples.trajectories) # (n_samples, T, act_dim+obs_dim)
			chains = to_np(samples.chains) # (n_samples, diffusion_T, T, act_dim+obs_dim)

			## [ n_samples x horizon x observation_dim ]
			normed_refs = refs[:, :, self.dynamic_cfg["act_dim"]:]
			normed_observations = trajectories[:, :, self.dynamic_cfg["act_dim"]:] # (n_samples, T, obs_dim)
			normed_chains = chains[:, :, :, self.dynamic_cfg["act_dim"]:] # (n_samples, horizon, T, obs_dim)

			## [ n_samples x (diffusion_T) x horizon + 1 x observation_dim ]
			observations = self.dynamic_cfg["dataset"].normalizer.unnormalize(normed_observations, 'observations') # [ n_samples x horizon x observation_dim ]
			chains = self.dynamic_cfg["dataset"].normalizer.unnormalize(normed_chains, 'observations') # [ n_samples x (diffusion_T) x horizon x observation_dim ]
			refs = self.dynamic_cfg["dataset"].normalizer.unnormalize(normed_refs, 'observations')

			## render
			ref_res.append(self.dynamic_cfg["dataset"].renderer.episodes2img(refs))
			img_res.append(self.dynamic_cfg["dataset"].renderer.episodes2img(observations))
			chain_res.append(self.dynamic_cfg["dataset"].renderer.chains2video(chains))
			# ref_res.append(self.dynamic_cfg["dataset"].renderer.episodes2img(refs,path=self.hparams+"/ref_latest.png"))
			# img_res.append(self.dynamic_cfg["dataset"].renderer.episodes2img(observations,path=self.hparams+"/ref_latest.png"))
			# chain_res.append(self.dynamic_cfg["dataset"].renderer.chains2video(chains,path=self.hparams+"/ref_latest.png"))
			

		return ref_res, img_res, None if len(chain_res) == 0 else chain_res

class DiffuserWithActModule(DefaultModule):
	def __init__(
		self,
		**kwargs
	):
		super().__init__(**kwargs)
		
	def step(self, batch: Any):
		""" process the batch from dataloader and return the res_batch
		input: `batch dict` from dataloader
		output: `res_batch` dict with "x", "y", "info", "outputs", "preds", "loss"
		ps. note that the calcultion of "outputs", "preds", "loss" could
			be task-specific, so we need to implement it in the subclass
		"""
		# s, s_, act
		kwargs = {}
		if len(batch) == 3: kwargs["valids"] = batch.valids
		loss, _ = self.net.loss(batch.trajectories, batch.conditions, **kwargs)
		# applying nosie robustness
		if self.hparams.data_noise: 
			batch = batch._replace(trajectories=batch.trajectories + torch.randn_like(batch.trajectories) * self.hparams.data_noise)
			for k, v in batch.conditions.items():
				batch.conditions[k] += torch.randn_like(v) * self.hparams.data_noise
		res_batch = {
			"outputs": batch.trajectories, # TODO
			"preds": batch.trajectories, # TODO
			"y": batch.trajectories, # TODO
			"loss": loss
		}
		return res_batch

	def validation_epoch_end(self, outputs):
		### init
		assert self.net.training == False, "net should be in eval mode"
		LOG_PREFIX = "val_ep_end"
		to_log = {}

		### get render data # TODO spilt well
		dataset = self.dynamic_cfg["dataset"]
		env = dataset.env
		# ! TODO uncomment the following
		ref_samples, img_samples, chain_samples = self.render_samples() # a [list of batch_size] with each one as one img but a composite one
		if self.hparams.controller.turn_on:
			to_log_ = eval_diffuser_witha(
				self, 
				self.hparams.controller.policy, 
				plan_freq=self.hparams.controller.plan_freq,
				guide_=self.hparams.controller.guide
			)
			to_log.update({"eval_pair/"+k: v for k, v in to_log_.items()})
		### log
		to_log["ref"] = [wandb_media_wrapper(_) for _ in ref_samples]
		if chain_samples is not None: 
			to_log["chain"] = [wandb_media_wrapper(_) for _ in chain_samples]
		to_log["samples"] = [wandb_media_wrapper(_) for _ in img_samples]
		LOG_PREFIX="val_ep_end"
		wandb.log({
			f"{LOG_PREFIX}/{k}": v for k, v in to_log.items()
		}, commit=True)
		super().validation_epoch_end(outputs)

	def render_samples(self):
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
		batch_size = 1
		N_SAMPLES = 4 # would have same condition, rendered in one img
		ref_res, img_res, chain_res = [], [], []
		dataset = self.dynamic_cfg["data_val"][0]
		from torch.utils.data.dataloader import default_collate
		from collections.abc import Mapping, Sequence

		def recursive_collate(batch):
			elem = batch[0]
			if isinstance(elem, Mapping):
				return {key: recursive_collate([d[key] for d in batch]) for key in elem}
			elif isinstance(elem, Sequence) and not isinstance(elem, str):
				return [recursive_collate(samples) for samples in zip(*batch)]
			else:
				return default_collate(batch)  # use PyTorch's default_collate

		for i in range(batch_size):
			
			## get a single datapoint
			batch = [dataset[random.randint(0, len(dataset)-1)]]
			batch = recursive_collate(batch) # stack another dim
			batch = EpisodeValidBatch(*batch) if len(batch) == 3 else EpisodeBatch(*batch)
			conditions = batch.conditions
			refs = to_np(batch.trajectories)

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
				'b ... -> (repeat b) ...', repeat=N_SAMPLES,
			)
			refs = einops.repeat(refs, 'b ... -> (repeat b) ...', repeat=N_SAMPLES)


			## [ n_samples x horizon x (action_dim + observation_dim) ]
			samples = self.net(conditions, return_chain=True) # ! ADD EMA in paper
			trajectories = to_np(samples.trajectories) # (n_samples, T, act_dim+obs_dim)
			chains = to_np(samples.chains) # (n_samples, diffusion_T, T, act_dim+obs_dim)

			## [ n_samples x horizon x observation_dim ]
			normed_refs = refs[:, :, self.dynamic_cfg["act_dim"]:]
			normed_observations = trajectories[:, :, self.dynamic_cfg["act_dim"]:] # (n_samples, T, obs_dim)
			normed_chains = chains[:, :, :, self.dynamic_cfg["act_dim"]:] # (n_samples, horizon, T, obs_dim)

			## [ n_samples x (diffusion_T) x horizon + 1 x observation_dim ]
			observations = self.dynamic_cfg["dataset"].normalizer.unnormalize(normed_observations, 'observations') # [ n_samples x horizon x observation_dim ]
			chains = self.dynamic_cfg["dataset"].normalizer.unnormalize(normed_chains, 'observations') # [ n_samples x (diffusion_T) x horizon x observation_dim ]
			refs = self.dynamic_cfg["dataset"].normalizer.unnormalize(normed_refs, 'observations')

			## render
			ref_res.append(self.dynamic_cfg["dataset"].renderer.episodes2img(refs))
			img_res.append(self.dynamic_cfg["dataset"].renderer.episodes2img(observations))
			chain_res.append(self.dynamic_cfg["dataset"].renderer.chains2video(chains))
			# ref_res.append(self.dynamic_cfg["dataset"].renderer.episodes2img(refs,path=self.hparams+"/ref_latest.png"))
			# img_res.append(self.dynamic_cfg["dataset"].renderer.episodes2img(observations,path=self.hparams+"/ref_latest.png"))
			# chain_res.append(self.dynamic_cfg["dataset"].renderer.chains2video(chains,path=self.hparams+"/ref_latest.png"))
			

		return ref_res, img_res, None if len(chain_res) == 0 else chain_res

class DiffuserBC(DefaultModule):
	def __init__(
		self,
		**kwargs
	):
		super().__init__(**kwargs)
		
	def step(self, batch: Any):
		""" process the batch from dataloader and return the res_batch
		input: `batch dict` from dataloader
		output: `res_batch` dict with "x", "y", "info", "outputs", "preds", "loss"
		ps. note that the calcultion of "outputs", "preds", "loss" could
			be task-specific, so we need to implement it in the subclass
		"""
		outputs = self.net(batch.s)
		res_batch = {
			"s": batch.s,
			"s_": batch.s_,
			"act": batch.act,
			# "info": batch["info"],
			"outputs": outputs,
			"preds": outputs,
			"loss": self.hparams.loss_func()(outputs, batch.act),
			"y": batch.act,
		}
		return res_batch

	def validation_epoch_end(self, outputs):
		### init
		assert self.net.training == False, "net should be in eval mode"
		LOG_PREFIX = "val_ep_end"
		to_log = {}

		### get render data # TODO spilt well
		dataset = self.dynamic_cfg["dataset"]
		env = dataset.env
		# ! TODO uncomment the following
		# ref_samples, img_samples, chain_samples = self.render_samples() # a [list of batch_size] with each one as one img but a composite one
		if self.hparams.controller.turn_on:
			to_log_ = eval_bc(self)
			to_log.update({"eval_pair/"+k: v for k, v in to_log_.items()})
		### log
		# to_log["ref"] = [wandb_media_wrapper(_) for _ in ref_samples]
		# if chain_samples is not None: 
		# 	to_log["chain"] = [wandb_media_wrapper(_) for _ in chain_samples]
		# to_log["samples"] = [wandb_media_wrapper(_) for _ in img_samples]
		LOG_PREFIX="val_ep_end"
		wandb.log({
			f"{LOG_PREFIX}/{k}": v for k, v in to_log.items()
		}, commit=True)
		super().validation_epoch_end(outputs)

	def render_samples(self):
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
		batch_size = 1
		N_SAMPLES = 4 # would have same condition, rendered in one img
		ref_res, img_res, chain_res = [], [], []
		dataset = self.dynamic_cfg["data_val"][0]
		from torch.utils.data.dataloader import default_collate
		from collections.abc import Mapping, Sequence

		def recursive_collate(batch):
			elem = batch[0]
			if isinstance(elem, Mapping):
				return {key: recursive_collate([d[key] for d in batch]) for key in elem}
			elif isinstance(elem, Sequence) and not isinstance(elem, str):
				return [recursive_collate(samples) for samples in zip(*batch)]
			else:
				return default_collate(batch)  # use PyTorch's default_collate

		for i in range(batch_size):
			
			## get a single datapoint
			batch = [dataset[random.randint(0, len(dataset)-1)]]
			batch = recursive_collate(batch) # stack another dim
			batch = EpisodeValidBatch(*batch) if len(batch) == 3 else EpisodeBatch(*batch)
			conditions = batch.conditions
			refs = to_np(batch.trajectories)

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
				'b ... -> (repeat b) ...', repeat=N_SAMPLES,
			)
			refs = einops.repeat(refs, 'b ... -> (repeat b) ...', repeat=N_SAMPLES)


			## [ n_samples x horizon x (action_dim + observation_dim) ]
			samples = self.net(conditions, return_chain=True) # ! ADD EMA in paper
			trajectories = to_np(samples.trajectories) # (n_samples, T, act_dim+obs_dim)
			chains = to_np(samples.chains) # (n_samples, diffusion_T, T, act_dim+obs_dim)

			## [ n_samples x horizon x observation_dim ]
			normed_refs = refs[:, :, self.dynamic_cfg["act_dim"]:]
			normed_observations = trajectories[:, :, self.dynamic_cfg["act_dim"]:] # (n_samples, T, obs_dim)
			normed_chains = chains[:, :, :, self.dynamic_cfg["act_dim"]:] # (n_samples, horizon, T, obs_dim)

			## [ n_samples x (diffusion_T) x horizon + 1 x observation_dim ]
			observations = self.dynamic_cfg["dataset"].normalizer.unnormalize(normed_observations, 'observations') # [ n_samples x horizon x observation_dim ]
			chains = self.dynamic_cfg["dataset"].normalizer.unnormalize(normed_chains, 'observations') # [ n_samples x (diffusion_T) x horizon x observation_dim ]
			refs = self.dynamic_cfg["dataset"].normalizer.unnormalize(normed_refs, 'observations')

			## render
			ref_res.append(self.dynamic_cfg["dataset"].renderer.episodes2img(refs))
			img_res.append(self.dynamic_cfg["dataset"].renderer.episodes2img(observations))
			chain_res.append(self.dynamic_cfg["dataset"].renderer.chains2video(chains))
			# ref_res.append(self.dynamic_cfg["dataset"].renderer.episodes2img(refs,path=self.hparams+"/ref_latest.png"))
			# img_res.append(self.dynamic_cfg["dataset"].renderer.episodes2img(observations,path=self.hparams+"/ref_latest.png"))
			# chain_res.append(self.dynamic_cfg["dataset"].renderer.chains2video(chains,path=self.hparams+"/ref_latest.png"))
			

		return ref_res, img_res, None if len(chain_res) == 0 else chain_res
