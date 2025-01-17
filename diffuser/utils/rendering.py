# Standard Libraries
import os
import warnings

# Third-Party Libraries
import numpy as np
import einops
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import gym
import mujoco_py as mjc
from typing import List, Dict, Union, Any
from copy import deepcopy
from src.func import *

# Local Application/Library Specific Imports
from .arrays import to_np
from .video import save_video, save_videos
from diffuser.datasets.d4rl import load_environment


#-----------------------------------------------------------------------------#
#------------------------------- helper structs ------------------------------#
#-----------------------------------------------------------------------------#

def env_map(env_name):
	'''
		map D4RL dataset names to custom fully-observed
		variants for rendering
	'''
	if 'halfcheetah' in env_name:
		return 'HalfCheetahFullObs-v2'
	if 'reacher' in env_name:
		return 'Reacher-v2'
	elif 'hopper' in env_name:
		return 'HopperFullObs-v2'
	elif 'walker2d' in env_name:
		return 'Walker2dFullObs-v2'
	elif 'kitchen' in env_name:
		return "FrankaKitchen-v1"
	elif 'door' in env_name:
		return "AdroitHandDoor-v1"
	elif 'hammer' in env_name:
		return "AdroitHandHammer-v1"
	elif 'pen' in env_name:
		return "AdroitHandPen-v1"
	elif 'relocate'in env_name:
		return "AdroitHandRelocate-v1"
	else:
		return env_name

#-----------------------------------------------------------------------------#
#------------------------------ helper functions -----------------------------#
#-----------------------------------------------------------------------------#

def atmost_2d(x):
	while x.ndim > 2:
		x = x.squeeze(0)
	return x

def zipsafe(*args):
	length = len(args[0])
	assert all([len(a) == length for a in args])
	return zip(*args)

def zipkw(*args, **kwargs):
	nargs = len(args)
	keys = kwargs.keys()
	vals = [kwargs[k] for k in keys]
	zipped = zipsafe(*args, *vals)
	for items in zipped:
		zipped_args = items[:nargs]
		zipped_kwargs = {k: v for k, v in zipsafe(keys, items[nargs:])}
		yield zipped_args, zipped_kwargs

def get_image_mask(img):
	background = (img == 255).all(axis=-1, keepdims=True) | (img == 0).all(axis=-1, keepdims=True) | (img == 12).all(axis=-1, keepdims=True)
	mask = ~background.repeat(3, axis=-1)
	return mask

def plot2img(fig, remove_margins=True):
	# https://stackoverflow.com/a/35362787/2912349
	# https://stackoverflow.com/a/54334430/2912349

	from matplotlib.backends.backend_agg import FigureCanvasAgg

	if remove_margins:
		fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

	canvas = FigureCanvasAgg(fig)
	canvas.draw()
	img_as_string, (width, height) = canvas.print_to_buffer()
	return np.fromstring(img_as_string, dtype='uint8').reshape((height, width, 4))

#-----------------------------------------------------------------------------#
#---------------------------------- renderers --------------------------------#
#-----------------------------------------------------------------------------#

class Renderer:
	"""
	A class containing common rendering functions.
	
	This class was designed to complement the Diffuser repository by providing simplified API functions for rendering.
	Although these methods rely on underlying code from the Diffuser repository, they offer a streamlined interface.
	Ideally, users should only need to interact with these API functions for rendering needs.
	"""
	def episodes2img(self, episodes: np.ndarray, cols: int = 1, path: Union[None, str] = None) -> np.ndarray:
		"""
		Renders an episode to an image. If there are multiple episodes, they are rendered in grid format.

		Args:
			episodes: A numpy array with shape (B, T, obs_dim) or (T, obs_dim).
			cols: Number of columns to use when rendering multiple episodes.
			path: File path to save the image. Currently unsupported, should be None.

		Returns:
			img: A numpy array with shape (H, W, C).
			
		Raises:
			ValueError: If episodes array is not 2 or 3 dimensional.
			AssertionError: If path is not None.
		"""
		
		if len(episodes.shape) == 3:
			episodes_list = [episodes[i] for i in range(len(episodes))]
			return self.composite(path, episodes_list, conditions={}, dim=(1024, 256))

		elif len(episodes.shape) == 2:
			render_kwargs = {
				'trackbodyid': 2,
				'distance': 10,
				'lookat': [5, 2, 0.5],
				'elevation': 0
			}
			return self.renders(to_np(path), dim=(1024, 256), partial=True, qvel=True, render_kwargs=render_kwargs)
		
		else:
			raise ValueError("episodes2img: episodes should be 2 or 3 dimensional")

	def chains2video(self, chains: np.ndarray, cols: int = 1, path: Union[None, str] = None) -> np.ndarray:
		"""
		Renders diffusion chains to video.

		Args:
			chains: A numpy array with shape (B, diff_T, T, obs_dim) or (diff_T, T, obs_dim).
			cols: Number of columns to use when rendering multiple chains.
			path: File path to save the video. Currently unsupported, should be None.

		Returns:
			video: A numpy array with shape (T, H, W, C).

		Raises:
			AssertionError: If path is not None or chains array is not 4-dimensional.
		"""
		INTERVAL = 0.2
		assert path is None, "path should be None"
		assert len(chains.shape) == 4, "chains should be 4-dimensional"

		B, diff_T, T, obs_dim = chains.shape
		interval = int(INTERVAL * diff_T) if isinstance(INTERVAL, float) else INTERVAL
		
		chain_frames = []
		
		for diff_t in range(0, diff_T, interval):
			chain_frames.append(self.episodes2img(chains[:, diff_t]))
			if len(chain_frames[0].shape) == 4: return chain_frames[0] #   ! DEBUG already a video

		video = np.stack(chain_frames, axis=0)  # T, H, W, Ch
		video = video.transpose(0, 3, 1, 2)  # T, Ch, H, W
		return video

	def composite(self, savepath, paths, conditions={}, dim=(1024, 256), ncol=1, **kwargs):
		raise NotImplementedError("composite: not implemented")

#-----------------------------------------------------------------------------#
#----------------------------------- maze2d ----------------------------------#
#-----------------------------------------------------------------------------#

MAZE_BOUNDS = {
	'maze2d-umaze-v1': (0, 5, 0, 5),
	'maze2d-medium-v1': (0, 8, 0, 8),
	'maze2d-large-v1': (0, 9, 0, 12),
	'maze2d-openlarge-v0': (0, 9, 0, 12),
	'maze2d-open-v0': (0, 7, 0, 7),
	'maze2d-open55-v0': (0, 7, 0, 7)
}

class MazeRenderer(Renderer):

	def __init__(self, env):
		if type(env) is str: env = load_environment(env)
		self._config = env._config
		self._background = self._config != ' '
		self._remove_margins = False
		self._extent = (0, 1, 1, 0)

	def renders(self, observations, conditions=None, title=None):
		plt.clf()
		fig = plt.gcf()
		fig.set_size_inches(5, 5)
		
		plt.imshow(self._background * .5,
			extent=self._extent, cmap=plt.cm.binary, vmin=0, vmax=1)

		path_length = len(observations)
		colors = plt.cm.jet(np.linspace(0,1,path_length))
		plt.plot(observations[:,1], observations[:,0], c='black', zorder=10)
		plt.scatter(observations[:,1], observations[:,0], c=colors, zorder=20)
		
		if conditions is not None:
			# plot a green at the start and red at the end
			# conditions = {0: np.array(4), path_length-1: np.array(4)}
			if 0 in conditions:
				plt.scatter(conditions[0][1], conditions[0][0], c='green', zorder=30, s=400, marker='o', edgecolors='black')
			for k, v in conditions.items():
				if k == 0: continue
				if k < 10: continue
				if type(k) == int:
					plt.scatter(v[1], v[0], c='red', zorder=30, s=400, marker='*',edgecolors='black')
					break
		
		plt.axis('off')
		plt.title(title)
		img = plot2img(fig, remove_margins=self._remove_margins)
		return img
	
	def render_to_gif(self, observation, savepath, conditions=None, title=None, **kwargs):
		# Define function for animation
		def update(i):
			plt.clf()
			fig = plt.gcf()
			plt.imshow(self._background * .5, extent=self._extent, cmap=plt.cm.binary, vmin=0, vmax=1)
			colors = plt.cm.jet(np.linspace(0,1,len(observation)))
			plt.plot(observation[i, 1], observation[i, 0], c='black', zorder=10)
			plt.scatter(observation[i, 1], observation[i, 0], c=[colors[i]], zorder=20)
			plt.axis('off')
			plt.title(title)

		# Create the animation object
		ani = animation.FuncAnimation(plt.gcf(), update, frames=len(observation), interval=200)

		# Save to file
		ani.save(savepath, writer='pillow')

	def composite(self, savepath, paths, conditions={}, dim=(1024, 256), ncol=2, **kwargs):
		'''
			savepath : str
			observations : [ n_paths x horizon x 2 ]
		'''
		assert len(paths) % ncol == 0, 'Number of paths must be divisible by number of columns'

		images_res = []
		# for path, kw in zipkw(paths, **kwargs):
			# img = self.renders(*path, conditions=conditions, **kw)
		for path in paths:
			img = self.renders(path, conditions=conditions)
			images_res.append(img)
		images = np.stack(images_res, axis=0)

		nrow = len(images) // ncol
		images = einops.rearrange(images,
			'(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol)
		if savepath is not None:
			imageio.imsave(savepath, images)
			print(f'Saved {len(paths)} samples to: {savepath}')

		# render only first sample to gif
		# self.render_to_gif(paths[0], savepath[:-4] + '.gif')
		return images
		
class Maze2dRenderer(MazeRenderer):

	def __init__(self, env, observation_dim=None):
		self.env_name = env
		self.env = load_environment(env)
		self.observation_dim = np.prod(self.env.observation_space.shape)
		self.action_dim = np.prod(self.env.action_space.shape)
		self.goal = None
		self._background = self.env.maze_arr == 10
		self._remove_margins = False
		self._extent = (0, 1, 1, 0)

	def renders(self, observations, conditions=None, **kwargs):
		bounds = MAZE_BOUNDS[self.env_name]

		observations = observations + 0.5
		# if "open55" in self.env.unwrapped.name: # ! dont know why we need to shift the obs to render
		#     observations -= 0.5
		# else:
			# observations += 0.5
		
		if len(bounds) == 2:
			_, scale = bounds
			observations /= scale
		elif len(bounds) == 4:
			_, iscale, _, jscale = bounds
			observations[:, 0] /= iscale
			observations[:, 1] /= jscale
		else:
			raise RuntimeError(f'Unrecognized bounds for {self.env_name}: {bounds}')

		if conditions is not None:
			conditions = conditions.copy()
			for k, v in conditions.items():
				conditions[k] = v + .5
			if len(bounds) == 2:
				for k, v in conditions.items():
					conditions[k] = v / scale
			elif len(bounds) == 4:
				for k, v in conditions.items():
					conditions[k][0] /= iscale
					conditions[k][1] /= jscale
			else:
				raise RuntimeError(f'Unrecognized bounds for {self.env_name}: {bounds}')
			
		return super().renders(observations, conditions, **kwargs)

	def render_rollout(self, savepath, states, conditions=None, **video_kwargs):
		if type(states) is list: states = np.array(states)
		image = self.renders(states, conditions=conditions)
		# save_video(savepath, images, **video_kwargs)
		imageio.imsave(savepath, image)
		print(f'Saved rollout to: {savepath}')
		return image

	def render_plan(self, savepath, actions, observations_pred, state, conditions=None, fps=30):
		## [ batch_size x horizon x observation_dim ]
		observations_real = rollouts_from_state(self.env, state, actions)

		## there will be one more state in `observations_real`
		## than in `observations_pred` because the last action
		## does not have an associated next_state in the sampled trajectory
		observations_real = observations_real[:,:-1]

		images_pred = np.stack([
			self._renders(obs_pred, partial=True)
			for obs_pred in observations_pred
		])

		images_real = np.stack([
			self._renders(obs_real, partial=False)
			for obs_real in observations_real
		])

		## [ batch_size x horizon x H x W x C ]
		images = np.concatenate([images_pred, images_real], axis=-2)
		save_videos(savepath, *images)

#-----------------------------------------------------------------------------#
#----------------------------------- mujoco ----------------------------------#
#-----------------------------------------------------------------------------#

class MuJoCoRenderer(Renderer):
	'''
		default mujoco renderer
	'''

	def __init__(self, env):
		if type(env) is str:
			env = env_map(env)
			import gym
			self.env = gym.make(env)
		else:
			self.env = env
		## - 1 because the envs in renderer are fully-observed
		if self.env.observation_space.shape is None:
			obs_shape = self.env.observation_space.spaces["observation"].shape
		else:
			obs_shape = self.env.observation_space.shape
		self.observation_dim = np.prod(obs_shape) - 1
		self.action_dim = np.prod(self.env.action_space.shape)
		try:
			self.viewer = mjc.MjRenderContextOffscreen(self.env.sim)
		except:
			print('[ utils/rendering ] Warning: could not initialize offscreen renderer')
			self.viewer = None

	def pad_observation(self, observation):
		state = np.concatenate([
			np.zeros(1),
			observation,
		])
		return state

	def pad_observations(self, observations):
		"""
		intergrad to get the position info
		"""
		qpos_dim = self.env.sim.data.qpos.size
		## xpos is hidden
		xvel_dim = qpos_dim - 1
		xvel = observations[:, xvel_dim]
		xpos = np.cumsum(xvel) * self.env.dt
		states = np.concatenate([
			xpos[:,None],
			observations,
		], axis=-1)
		return states

	def render(self, observation, dim=256, partial=False, qvel=True, render_kwargs=None, conditions=None):
		
		if type(dim) == int:
			dim = (dim, dim)

		if self.viewer is None:
			return np.zeros((*dim, 3), np.uint8)

		if render_kwargs is None:
			xpos = observation[0] if not partial else 0
			render_kwargs = {
				'trackbodyid': 2,
				'distance': 3,
				'lookat': [xpos, -0.5, 1],
				'elevation': -20
			}

		# ! only for reacher since it does not move
		if "reacher" in self.env.unwrapped.spec.id.lower():
			render_kwargs = {'elevation': 90}

		for key, val in render_kwargs.items():
			if key == 'lookat':
				self.viewer.cam.lookat[:] = val[:]
			else:
				setattr(self.viewer.cam, key, val)

		if partial:
			state = self.pad_observation(observation)
		else:
			state = observation

		qpos_dim = self.env.sim.data.qpos.size
		if not qvel or state.shape[-1] == qpos_dim:
			qvel_dim = self.env.sim.data.qvel.size
			state = np.concatenate([state, np.zeros(qvel_dim)])

		set_state(self.env, state)

		self.viewer.render(*dim)
		data = self.viewer.read_pixels(*dim, depth=False)
		data = data[::-1, :, :]

		# data shape=(h,w,ch), value=0-255
		# save to debug/img.png
		# DEBUG = True
		# if DEBUG:
		#     from PIL import Image
		#     import numpy as np
		#     image = Image.fromarray(np.uint8(data))
		#     image.save('debug/img.png')

		return data

	def _renders(self, observations, **kwargs):
		images = []
		for observation in observations:
			img = self.render(observation, **kwargs)
			images.append(img)
		return np.stack(images, axis=0)

	def renders(self, samples, partial=False, **kwargs):        
		if partial: # True for mujoco, can add position
			samples = self.pad_observations(samples)
			partial = False
		samples_ori = samples


		samples = samples_ori
		RENDER_FREQ = 16 # only render evey n steps
		COLOR_BASE_NUM = 10
		if "hopper" in self.env.unwrapped.spec.id.lower():
			RENDER_FREQ = 8
		elif "cheetah" in self.env.unwrapped.spec.id.lower():
			RENDER_FREQ = 4
		elif "walker2d" in self.env.unwrapped.spec.id.lower():
			RENDER_FREQ = 8
		elif "reacher" in self.env.unwrapped.spec.id.lower():
			COLOR_BASE_NUM = 40
			RENDER_FREQ = 8
		
		samples = samples[::RENDER_FREQ]
		sample_images = self._renders(samples, partial=partial, **kwargs)
		composite = np.ones_like(sample_images[0]) * 255

		# rainbow colors
		color_base = plt.cm.jet(np.linspace(0,1,COLOR_BASE_NUM))[:,:3] # [N, 3]
		# colors = np.tile(color_base, (len(samples) // len(color_base)) + 1, axis=0)[:len(samples)]
		colors = np.concatenate([color_base for _ in range((len(samples) // len(color_base)) +1)], axis=0)[:len(samples)]
		for i, img in enumerate(sample_images): # [H, W, 3]
			mask = get_image_mask(img)
			# mask = np.ones_like(img)
			color_board = np.ones_like(img)
			color_board[:,:,0] = colors[i,0] * 255
			color_board[:,:,1] = colors[i,1] * 255
			color_board[:,:,2] = colors[i,2] * 255
			composite[mask] = color_board[mask]
		import imageio
		imageio.imsave("./debug/render.png", composite)
		return composite

	def composite(self, savepath, paths, conditions={}, dim=(4096, 256), ncol=1, **kwargs):

		render_kwargs = {
			'trackbodyid': 2,
			'distance': 6,
			'lookat': [5, 2, 0.5],
			'elevation': 0
		}
		images = []
		for path in paths:
			## [ H x obs_dim ]
			path = atmost_2d(path)
			img = self.renders(to_np(path), dim=dim, partial=True, qvel=True, render_kwargs=render_kwargs, **kwargs)
			images.append(img)
		images = np.concatenate(images, axis=0)

		if savepath is not None:
			imageio.imsave(savepath, images)
			print(f'Saved {len(paths)} samples to: {savepath}')

		return images

	def render_rollout(self, savepath, states, conditions={}, **video_kwargs):
		if type(states) is list: states = np.array(states)
		images = self._renders(states, partial=True)
		save_video(savepath, images, **video_kwargs)
		return images[0]

	def render_plan(self, savepath, actions, observations_pred, state, fps=30):
		## [ batch_size x horizon x observation_dim ]
		observations_real = rollouts_from_state(self.env, state, actions)

		## there will be one more state in `observations_real`
		## than in `observations_pred` because the last action
		## does not have an associated next_state in the sampled trajectory
		observations_real = observations_real[:,:-1]

		images_pred = np.stack([
			self._renders(obs_pred, partial=True)
			for obs_pred in observations_pred
		])

		images_real = np.stack([
			self._renders(obs_real, partial=False)
			for obs_real in observations_real
		])

		## [ batch_size x horizon x H x W x C ]
		images = np.concatenate([images_pred, images_real], axis=-2)
		save_videos(savepath, *images)
		return images[0]

	def render_diffusion(self, savepath, diffusion_path, **video_kwargs):
		'''
			diffusion_path : [ n_diffusion_steps x batch_size x 1 x horizon x joined_dim ]
		'''
		render_kwargs = {
			'trackbodyid': 2,
			'distance': 10,
			'lookat': [10, 2, 0.5],
			'elevation': 0,
		}

		diffusion_path = to_np(diffusion_path)

		n_diffusion_steps, batch_size, _, horizon, joined_dim = diffusion_path.shape

		frames = []
		for t in reversed(range(n_diffusion_steps)):
			print(f'[ utils/renderer ] Diffusion: {t} / {n_diffusion_steps}')

			## [ batch_size x horizon x observation_dim ]
			states_l = diffusion_path[t].reshape(batch_size, horizon, joined_dim)[:, :, :self.observation_dim]

			frame = []
			for states in states_l:
				img = self.composite(None, states, dim=(1024, 256), partial=True, qvel=True, render_kwargs=render_kwargs)
				frame.append(img)
			frame = np.concatenate(frame, axis=0)

			frames.append(frame)

		save_video(savepath, frames, **video_kwargs)

	def __call__(self, *args, **kwargs):
		return self.renders(*args, **kwargs)

class MarinaRenderer(MuJoCoRenderer):
	def __init__(self, env):
		if type(env) is str:
			env = env_map(env)
			import gymnasium as gym
			self.env = gym.make(env, render_mode="rgb_array")
		else:
			self.env = env
		## - 1 because the envs in renderer are fully-observed
		if self.env.observation_space.shape is None:
			obs_shape = self.env.observation_space.spaces["observation"].shape
		else:
			obs_shape = self.env.observation_space.shape
		self.observation_dim = np.prod(obs_shape)
		self.action_dim = np.prod(self.env.action_space.shape)
		self.env.reset()

	def render(self, observation, dim=256, partial=False, qvel=True, render_kwargs=None, conditions=None):

		if type(dim) == int:
			dim = (dim, dim)

		# if self.viewer is None:
		#     return np.zeros((*dim, 3), np.uint8)

		# if render_kwargs is None:
		#     xpos = observation[0] if not partial else 0
		#     render_kwargs = {
		#         'trackbodyid': 2,
		#         'distance': 3,
		#         'lookat': [xpos, -0.5, 1],
		#         'elevation': -20
		#     }

		# for key, val in render_kwargs.items():
		#     if key == 'lookat':
		#         self.viewer.cam.lookat[:] = val[:]
		#     else:
		#         setattr(self.viewer.cam, key, val)

		if partial:
			state = self.pad_observation(observation)
		else:
			state = observation

		# qpos_dim = self.env.sim.data.qpos.size
		# if not qvel or state.shape[-1] == qpos_dim:
		#     qvel_dim = self.env.sim.data.qvel.size
		#     state = np.concatenate([state, np.zeros(qvel_dim)])

		# ! DEBUG
		# self.env.reset()
		# s = self.env.step(self.env.action_space.sample())[0]
		# qs = self.env.get_env_state()
		# qpos = self.env.get_env_state()['qpos']

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
		self.env.set_env_state(qs_cur)

		img = self.env.render() # h, w, 3
		# # save img to ./debug/test.png
		# import imageio
		# imageio.imsave("./debug/test.png", img)
		# print("saved img to ./debug/test.png")
		# sample to 50,50,3

		### ! DEBUG
		# import imageio
		# for idx in range(len(qs_cur["qpos"]))[-3:]:
		#     for value in [-0.75,-.5,-0.25,0.0,0.25,0.5,.75,1.0]:
		#         qtmp = deepcopy(qs_cur)
		#         qtmp["qpos"][idx] = value
		#         self.env.set_env_state(qtmp)
		#         img = self.env.render() # h, w, 3
		#         if not os.path.exists("./debug"): os.makedirs("./debug")
		#         if not os.path.exists(f"./debug/{idx}"): os.makedirs(f"./debug/{idx}")
		#         imageio.imsave(f"./debug/{idx}/test{value}.png", img)
		#         print(f"./debug/{idx}/test{value}.png")

		# # save img to ./debug/test.png
		# import imageio
		# imageio.imsave("./debug/test.png", img)
		# print("saved img to ./debug/test.png")
		# sample to 50,50,3



		img = img[::2, ::2, :]
		return img

	def pad_observations(self, observations):
		# cur_state = self.env.unwrapped.get_env_state()
		# qpos_dim = cur_state["qvel"].size
		# # qpos_dim = self.env.sim.data.qpos.size
		# ## xpos is hidden
		# xvel_dim = qpos_dim - 1
		# xvel = observations[:, xvel_dim]
		# xpos = np.cumsum(xvel) * self.env.dt
		# states = np.concatenate([
		#     xpos[:,None],
		#     observations,
		# ], axis=-1)
		return observations

	def renders(self, observations, **kwargs):
		# origin fro diffuser (render n images)
		images = [] 
		for observation in observations:
			img = self.render(observation, **kwargs)
			images.append(img)
		# return np.stack(images, axis=0)
		images = np.concatenate(images, axis=1)


		# if partial:
		#     samples = self.pad_observations(samples)
		#     partial = False

		# sample_images = self._renders(observations, partial=partial, **kwargs)

		# composite = np.ones_like(sample_images[0]) * 255

		# for img in sample_images:
		#     mask = get_image_mask(img)
		#     composite[mask] = img[mask]

		return images

	def renders_video(self, observations, **kwargs):
		# origin fro diffuser (render n images)
		images = [] 
		for observation in observations:
			img = self.render(observation, **kwargs)
			images.append(img)
		# return np.stack(images, axis=0)
		images = np.stack(images, axis=0) # T, H, W, C
		return images
	
	def composite(self, savepath, paths, conditions={}, dim=(1024, 256), ncol=1, **kwargs):
		images_res = []
		# for path, kw in zipkw(paths, **kwargs):
			# img = self.renders(*path, conditions=conditions, **kw)
		for path in paths:
			img = self.renders_video(path) # T, H, W, C
			images_res.append(img)
		images = np.stack(images_res, axis=0) # N, T, H, W, C
		
		nrow = len(images) // ncol
		# to video # (T, Ch, H, W) stack 4 in one
		# images = einops.rearrange(images,
		#     '(nrow ncol) T H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol)
		images = einops.rearrange(images, 
			'(nrow ncol) T H W Ch -> T Ch (nrow H) (ncol W)', nrow=nrow, ncol=ncol
		)

		# if savepath is not None:
		#     imageio.imsave(savepath, images)
		#     print(f'Saved {len(paths)} samples to: {savepath}')
		
		return images

class QuickdrawRenderer(Renderer):
	def __init__(self):
		self._remove_margins = False

	def renders(self, observations, conditions=None, title=None):
		plt.clf()
		fig = plt.gcf()
		fig.set_size_inches(5, 5)
		
		# plt.imshow(self._background * .5,
		# 	extent=self._extent, cmap=plt.cm.binary, vmin=0, vmax=1)

		s = observations
		for i in range(len(s)):
			if s[i][2] == 1:
				if i+2 >= len(s): continue
				plt.plot(s[i:i+2, 0], s[i:i+2, 1], color='red')
			elif s[i][2] == 0: 
				if i+2 >= len(s): continue
				plt.plot(s[i:i+2, 0], s[i:i+2, 1], color='blue')
		plt.xlim(0, 256)
		plt.ylim(0, 256)

		## ! DEBUG save path
		# import os
		# output_path = "./output" + '/quickdraw'
		# plt.savefig(output_path + '/' + 'img.png')
		# print(output_path)
		# if not os.path.exists(output_path): os.makedirs(output_path)
		
		# path_length = len(observations)
		# colors = plt.cm.jet(np.linspace(0,1,path_length))
		# plt.plot(observations[:,1], observations[:,0], c='black', zorder=10)
		# plt.scatter(observations[:,1], observations[:,0], c=colors, zorder=20)
		
		# if conditions is not None:
		# 	# plot a green at the start and red at the end
		# 	# conditions = {0: np.array(4), path_length-1: np.array(4)}
		# 	if 0 in conditions:
		# 		plt.scatter(conditions[0][1], conditions[0][0], c='green', zorder=30, s=400, marker='o', edgecolors='black')
		# 	for k, v in conditions.items():
		# 		if k == 0: continue
		# 		if k < 10: continue
		# 		if type(k) == int:
		# 			plt.scatter(v[1], v[0], c='red', zorder=30, s=400, marker='*',edgecolors='black')
		# 			break
		
		# set x,y lim to [0,256]

		plt.axis('off')
		plt.title(title)
		img = plot2img(fig, remove_margins=self._remove_margins)
		return img

	def composite(self, savepath, paths, conditions={}, dim=(1024, 256), ncol=1, **kwargs):
		'''
			savepath : str
			observations : [ n_paths x horizon x 2 ]
		'''
		assert len(paths) % ncol == 0, 'Number of paths must be divisible by number of columns'

		images_res = []
		# for path, kw in zipkw(paths, **kwargs):
			# img = self.renders(*path, conditions=conditions, **kw)
		for path in paths:
			img = self.renders(path, conditions=conditions)
			images_res.append(img)
		images = np.stack(images_res, axis=0)

		nrow = len(images) // ncol
		images = einops.rearrange(images,
			'(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol)
		if savepath is not None:
			imageio.imsave(savepath, images)
			print(f'Saved {len(paths)} samples to: {savepath}')

		# render only first sample to gif
		# self.render_to_gif(paths[0], savepath[:-4] + '.gif')
		return images

class PandaRenderer(Renderer):
	RENDER_MODE = "video" # video or image

	def __init__(self, env):
		if type(env) is str:
			self.env = gym_make_panda(env)
		else:
			raise ValueError(f'Expected str, got {type(env)}, this would cause env conflicts')
		self.env.reset()
		self.composite = self.composite_video if self.RENDER_MODE == "video" else self.composite_img

	def composite_img(self, savepath, paths, conditions={}, dim=(1024, 256), ncol=1, **kwargs):
		'''
			savepath : str
			observations : [ n_paths x horizon x 2 ]
		'''
		assert len(paths) % ncol == 0, 'Number of paths must be divisible by number of columns'

		images_res = []
		# for path, kw in zipkw(paths, **kwargs):
			# img = self.renders(*path, conditions=conditions, **kw)
		for path in paths:
			img = self.renders_img(path, conditions=conditions)
			images_res.append(img)
		images = np.stack(images_res, axis=0)

		nrow = len(images) // ncol
		images = einops.rearrange(images,
			'(nrow ncol) H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol)
		if savepath is not None:
			imageio.imsave(savepath, images)
			print(f'Saved {len(paths)} samples to: {savepath}')

		# render only first sample to gif
		# self.render_to_gif(paths[0], savepath[:-4] + '.gif')
		return images

	def composite_video(self, savepath, paths, conditions={}, dim=(1024, 256), ncol=1, **kwargs):
		images_res = []
		# for path, kw in zipkw(paths, **kwargs):
			# img = self.renders(*path, conditions=conditions, **kw)
		for path in paths:
			img = self.renders_video(path) # T, H, W, C
			images_res.append(img)
		images = np.stack(images_res, axis=0) # N, T, H, W, C
		
		nrow = len(images) // ncol
		# to video # (T, Ch, H, W) stack 4 in one
		# images = einops.rearrange(images,
		#     '(nrow ncol) T H W C -> (nrow H) (ncol W) C', nrow=nrow, ncol=ncol)
		images = einops.rearrange(images, 
			'(nrow ncol) T H W Ch -> T Ch (nrow H) (ncol W)', nrow=nrow, ncol=ncol
		)

		# if savepath is not None:
		#     imageio.imsave(savepath, images)
		#     print(f'Saved {len(paths)} samples to: {savepath}')
		
		return images

	def renders_img(self, observations, conditions=None, title=None):
		frames = draw_3d_path({
			"goal": (self.env.obs_handler.distill(observations,"goal"), "red"),
			"endpoint": (self.env.obs_handler.distill(observations,"endpoint"), "green"),
			"achieved": (self.env.obs_handler.distill(observations,"achieved"), "blue"),
		})
		# imageio.imsave("./debug/1123.png", frames)
		# print(f'Saved samples to: ./debug/1123.png')
		return frames

	def renders_video(self, observations, conditions=None, title=None):
		# ! DEBUG
		# observations = observations[:4]
		frames = []
		for obs in observations:
			self.env.set_state(obs)
			frame = self.env.render()
			frames.append(frame) # (h, w, 3)
		return np.stack(frames, axis=0) # (T,h,w,3)


def draw_3d_path(lines):
	"""
	Draw lines in 3D space and return as a NumPy image array.

	Parameters:
	lines: Each numpy array represents a line in 3D space, shaped (T, 3).
		e.g. {
			"goal": ((T,3), "green"),
			"endpoint": ((T,3), "red"),
			"achieved": ((T,3), "blue"),
		}
	colors (list of colors): Optional. A list of colors for each line. Must be the same length as lines if specified.
	"""
	from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
	   # Create a new 3D plot
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	# Iterate over each line
	for name, (line, color) in lines.items():
		# Extract x, y, z coordinates
		x, y, z = line[:, 0], line[:, 1], line[:, 2]
		# Plot the line
		ax.plot(x, y, z, color=color, label=name, alpha=1.0, zorder=-1)
		# plot gradient dot line to see trend
		for i in range(len(x)-1):
			# ax.plot([x[i], x[i+1]], [y[i], y[i+1]], [z[i], z[i+1]], color=color, alpha=i/len(x), linewidth=5)
			ax.scatter(x[i], y[i], z[i], color=color, alpha=i/len(x))
			# plot a bal
			# ax.plot_surface(x[i], y[i], z[i], color=color, alpha=i/len(x))
		if name == "goal":
			# plot big star sequence
			ax.plot(x, y, z, color=color, marker='*', markersize=10, zorder=100)
	# legend
	ax.legend()
	# Set labels for axes
	ax.set_xlabel('X axis')
	ax.set_ylabel('Y axis')
	ax.set_zlabel('Z axis')
	# lim -10., 10.
	LIM = 5.
	ax.set_xlim(-LIM, LIM)
	ax.set_ylim(-LIM, LIM)
	ax.set_zlim(-LIM*.5, LIM*.5)
	ax.set_box_aspect((1, 1, 0.5))
	# Convert the Matplotlib figure to a NumPy array
	canvas = FigureCanvas(fig)
	canvas.draw()
	image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
	image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	# Close the figure
	plt.close(fig)
	return image

#-----------------------------------------------------------------------------#
#---------------------------------- rollouts ---------------------------------#
#-----------------------------------------------------------------------------#

def set_state(env, state):
	qpos_dim = env.sim.data.qpos.size
	qvel_dim = env.sim.data.qvel.size
	if not state.size == qpos_dim + qvel_dim:
		warnings.warn(
			f'[ utils/rendering ] Expected state of size {qpos_dim + qvel_dim}, '
			f'but got state of size {state.size}')
		state = state[:qpos_dim + qvel_dim]

	env.set_state(state[:qpos_dim], state[qpos_dim:])

def rollouts_from_state(env, state, actions_l):
	rollouts = np.stack([
		rollout_from_state(env, state, actions)
		for actions in actions_l
	])
	return rollouts

def rollout_from_state(env, state, actions):
	qpos_dim = env.sim.data.qpos.size
	env.set_state(state[:qpos_dim], state[qpos_dim:])
	observations = [env._get_obs()]
	for act in actions:
		obs, rew, term, _ = env.step(act)
		observations.append(obs)
		if term:
			break
	for i in range(len(observations), len(actions)+1):
		## if terminated early, pad with zeros
		observations.append( np.zeros(obs.size) )
	return np.stack(observations)
