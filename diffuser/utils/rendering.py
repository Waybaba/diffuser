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
    elif 'hopper' in env_name:
        return 'HopperFullObs-v2'
    elif 'walker2d' in env_name:
        return 'Walker2dFullObs-v2'
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
    background = (img == 255).all(axis=-1, keepdims=True)
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
            self.env = gym.make(env)
        else:
            self.env = env
        ## - 1 because the envs in renderer are fully-observed
        self.observation_dim = np.prod(self.env.observation_space.shape) - 1
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
        return data

    def _renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def renders(self, samples, partial=False, **kwargs):
        if partial:
            samples = self.pad_observations(samples)
            partial = False

        sample_images = self._renders(samples, partial=partial, **kwargs)

        composite = np.ones_like(sample_images[0]) * 255

        for img in sample_images:
            mask = get_image_mask(img)
            composite[mask] = img[mask]

        return composite

    def composite(self, savepath, paths, conditions={}, dim=(1024, 256), ncol=1, **kwargs):

        render_kwargs = {
            'trackbodyid': 2,
            'distance': 10,
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
