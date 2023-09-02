import os
import numpy as np
import einops
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import gym
import mujoco_py as mjc
import warnings
import pdb

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

#-----------------------------------------------------------------------------#
#----------------------------------- maze2d ----------------------------------#
#-----------------------------------------------------------------------------#

MAZE_BOUNDS = {
    'maze2d-umaze-v1': (0, 5, 0, 5),
    'maze2d-medium-v1': (0, 8, 0, 8),
    'maze2d-large-v1': (0, 9, 0, 12),
    'maze2d-openlarge-v0': (0, 9, 0, 12)
}

class MazeRenderer:

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
                    # plt.scatter(conditions[path_length-1][1], conditions[path_length-1][0], c='red', zorder=30, s=600, marker='*',edgecolors='black')
                    plt.scatter(v[1], v[0], c='red', zorder=30, s=400, marker='*',edgecolors='black')
                    break
        
        plt.axis('off')
        plt.title(title)
        img = plot2img(fig, remove_margins=self._remove_margins)
        return img

    # def _renders(self, observations, **kwargs):
    #     images = []
    #     for observation in observations:
    #         img = self.renders(observation, **kwargs)
    #         images.append(img)
    #     return np.stack(images, axis=0)
    
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

    def composite(self, savepath, paths, conditions=None, ncol=2, **kwargs):
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

        observations = observations + .5
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

    def episodes2img(self, episodes, cols=1, path=None):
        """ rendering episode to img
        if multiple, render in grid format
        # Args:
            episode: (B, T, obs_dim) or (T, obs_dim)
            cols: used when rendering multiple episodes
        # Returns:
            img: (H, W, C)
        """
        assert path is None, "path is not None"
        if len(episodes.shape) == 3: 
            episodes = [episodes[i] for i in range(len(episodes))]
            return self.composite(path, episodes, conditions={}, dim=(1024, 256))
        elif len(episodes.shape) == 2:
            render_kwargs = {
                'trackbodyid': 2,
                'distance': 10,
                'lookat': [5, 2, 0.5],
                'elevation': 0
            }
            return self.renders(to_np(path), dim=(1024, 256), partial=True, qvel=True, render_kwargs=render_kwargs)
        else:
            raise ValueError("episodes2img: episodes should be 2 or 3 dim")

    def chains2video(self, chains, cols=1, path=None):
        """
        # Args:
            chains: (B, diff_T, T, obs_dim) or (diff_T, T, obs_dim)
            cols: used when rendering multiple chains
        # Implementation:
            call episodes2img then make gif
        # Returns:
            video: (T, H, W, C)
        """
        INTERVAL = 0.2
        assert path is None, "path is not None"
        assert len(chains.shape) == 4, "chains should be 4 dim"
        B, diff_T, T, obs_dim = chains.shape
        if isinstance(INTERVAL, float): INTERVAL = int(INTERVAL*diff_T)
        chain_frames = []
        for diff_t in range(0, diff_T, INTERVAL):
            chain_frames.append(self.episodes2img(chains[:,diff_t]))
            # for t in range(T):
                # frame = chain_b[t].transpose(1, 2, 0)
                # frames.append(frame)
            # if path: imageio.mimsave(path, frames, 'GIF', duration=2.0/len(frames))
        return np.stack(chain_frames, axis=0)
        

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
