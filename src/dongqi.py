import numpy as np
from copy import deepcopy
import imageio

def minari_to_d4rl(dataset):
	# ["observations", "actions", "terminals", "timeouts", "rewards"]
	ep_list = []
	for ep in dataset.iterate_episodes():
		ep_list.append(ep)
		break
	res = {}
	if isinstance(ep_list[0].observations, dict):
		res["observations"] = np.concatenate([ep.observations["observation"] for ep in ep_list], axis=0)
	else:
		res["observations"] = np.concatenate([ep.observations for ep in ep_list], axis=0)
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
	def render(self, observations):
		for step in range(0, observations.shape[0], 10):
			observation = observations[step]
			qs_cur = self.env.get_env_state()
			qs_cur = deepcopy(qs_cur)
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
					# observation[q_dim-1+1:q_dim+2], # ? seem to be some position
					np.array([0.]),
					observation[:q_dim-3],
					observation[q_dim-2:q_dim-1], # door pos
					observation[q_dim-3:q_dim-2], # latch pos
				], axis=0)
				# qs_cur["door_body_pos"] = observation[q_dim-1+3:q_dim-1+3+3]
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

			imageio.imsave("./debug/test_{}.png".format(step), img)
			print("saved img to ./debug/test.png")


if __name__ == "__main__":
	env, dataset = load_minari("door-human-v1") # e.g. hammer-human-v1, door-human-v1
	observations = dataset["observations"][:200]
	renderer = MinariRenderer(env)
	renderer.render(observations)
	

	