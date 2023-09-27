import numpy as np
from copy import deepcopy
import imageio
import pickle

def minari_to_d4rl(dataset):
	# ["observations", "actions", "terminals", "timeouts", "rewards"]
	ep_list = []

	a = []
	# for ep in dataset.iterate_episodes():
	# 	# print(ep.observations[0, 32:35])
	# 	a.append(ep.observations[0, 32:35])
	# print(np.mean(a, axis=0))

	for i, ep in enumerate(dataset.iterate_episodes()):
		
		ep_list.append(ep)
		if i == 5:
			break
		
	res = {}
	if isinstance(ep_list[0].observations, dict):
		res["observations"] = np.concatenate([ep.observations["observation"][:-1] for ep in ep_list], axis=0)
	else:
		res["observations"] = np.concatenate([ep.observations[:-1] for ep in ep_list], axis=0)
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

class MinariRenderer:
	def __init__(self, env):
		self.env = env
		self.door_body_xyz = None
	def render(self, observations, timeouts):
		for step in range(0, observations.shape[0], 5):
			print(step, timeouts[step])
			observation = observations[step]\

			# if observation[28] > 1: # finished
			# 	continue

			qs_cur = self.env.get_env_state()
			qs_cur = deepcopy(qs_cur)
			# print(qs_cur)
			# print(observation[28])
			q_dim = qs_cur["qpos"].shape[0]
			if 'pen' in self.env.unwrapped.spec.id.lower():
				qs_cur["qpos"][:-6] = observation[:q_dim-6] # qp[:-6]
				qs_cur["qpos"][-6:-3] = observation[q_dim-6:q_dim-3]
				qs_cur["qpos"][-3:] = observation[q_dim:q_dim+3]
			elif 'hammer' in self.env.unwrapped.spec.id.lower(): # 33
				qs_cur["qpos"][:-6] = observation[:q_dim-6] # qp[:-6]
				# qs_cur["qpos"][-6:-3] = observation[q_dim:q_dim+3]#  ! TODO
				# qs_cur["qpos"][-3:] = observation[q_dim+3:q_dim+6]#  ! TODO 
				# qs_cur["qpos"][-6:] = observation[q_dim+3:q_dim+9]
				qs_cur["qpos"][-6+0] = observation[q_dim+3+1]
				qs_cur["qpos"][-6+1] = observation[q_dim+3+0]
				qs_cur["qpos"][-6+2] = observation[q_dim+3+2]
				qs_cur["qpos"][-6+0+3] = observation[q_dim+3+0+3]
				qs_cur["qpos"][-6+1+3] = observation[q_dim+3+1+3]
				qs_cur["qpos"][-6+2+3] = observation[q_dim+3+2+3]
				# [26,27)       [33,36)     [36,42)         [42,45)
				# nail_inser    palm_xyz    hammer_xyzrad   nail_xyz
				# qs_cur['board_pos']
				qs_cur['board_pos'] = observation[q_dim+9:q_dim+12]
			elif 'door' in self.env.unwrapped.spec.id.lower():
				qs_cur["qpos"] = np.concatenate([ # 30
					# Dongqi: qpos的第0维是这样的
					np.array([np.sqrt(np.sum(observation[29:32]**2))]), 
					observation[:27],
					observation[28:29], # door_hinge pos
					observation[27:28]  # latch pos
				], axis=0)

				# Dongqi: 在每个episode开始的时候，把qs_cur["door_body_pos"]这样设定，在这个episode里保持不动
				if step % 200 == 0:
					self.door_body_xyz = observation[32:35]
					# print(observation[32:35])
				qs_cur["door_body_pos"] = np.array([self.door_body_xyz[0] - 0.29, self.door_body_xyz[1] + 0.15, self.door_body_xyz[2] + 0.025])

				# [29,32)       [32,35)
				# palm_xyz      handle_xyz
			elif 'relocate' in self.env.unwrapped.spec.id.lower():
				qs_cur["qpos"][:q_dim-6] = observation[:q_dim-6]
				palm_minus_obj = observation[q_dim-6:q_dim-3]
				palm_minus_tgt = observation[q_dim-3:q_dim]
				obj_minus_tgt = observation[q_dim:q_dim+3]
				# qs_cur["qpos"][q_dim-6:q_dim-3] no change
				qs_cur["qpos"][q_dim-3:q_dim] = - palm_minus_obj + qs_cur["qpos"][q_dim-6:q_dim-3]
			else:
				raise NotImplementedError(f"The mapping from s to qpos is not implemented yet. {self.env}")
			# qs_cur["qpos"] = np.concatenate([observation[:29], np.array([0.0])], axis=0)
			if "target_pos" in qs_cur: del qs_cur["target_pos"]
			if "hand_pos" in qs_cur: del qs_cur["hand_pos"]
			
			self.env.reset()
			
			self.env.set_env_state(qs_cur)
			
			img = self.env.render() # h, w, 3

			imageio.imsave("./debug/hammer/test_%3.3d.png" % step, img)

		print("saved img to ./debug/hammer/")


if __name__ == "__main__":
	env, dataset = load_minari("hammer-expert-v1") # e.g. hammer-human-v1, door-human-v1
	observations = dataset["observations"][:100]
	timeouts = dataset["timeouts"][:100]

	# a = np.zeros([1000, 3])
	# for kk in range(1000):
	# 	env.reset()
	# 	a[kk, :] = env.get_env_state()["door_body_pos"]

	# print(np.mean(a, axis=0))

	# # pickle.dump(observations, open("./debug/observations.pkl", "wb"))
	# # pickle.dump(env, open("./debug/env.pkl", "wb"))

	# env = pickle.load(open("./debug/env.pkl", "rb"))
	# observations = pickle.load(open("./debug/observations.pkl", "rb"))
	
	renderer = MinariRenderer(env)
	renderer.render(observations, timeouts)
	

	