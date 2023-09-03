import wandb

from gym.envs.registration import register

import hydra
from omegaconf import OmegaConf
from pathlib import Path

import numpy as np
import torch

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



class TrainDiffuserRunner:
    
    def start(self, cfg):
        print("Running default runner")
        self.cfg = cfg

        self.datamodule = cfg.datamodule()
        self.modelmodule = cfg.modelmodule(
            dataset_info=self.datamodule.info,
        )
        trainer = cfg.trainer(
            callbacks=[v for k,v in cfg.callbacks.items()],
            logger=[v for k,v in cfg.logger.items()],
        )
        trainer.fit(
            model=self.modelmodule,
            datamodule=self.datamodule,
        )
        print("Finished!")
        exit()


        ### ! TODO remove
        dataset = cfg.dataset()
        render = cfg.render(dataset.env.name)
        
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


        # save diffuser training cfg for inference reload
        

        ### train

        n_epochs = int(cfg.global_cfg.n_train_steps // cfg.global_cfg.n_steps_per_epoch)
        for epoch in range(n_epochs):
            print(f'Epoch {epoch} / {n_epochs} | {cfg.output_dir}')
            trainer.train(n_train_steps=cfg.global_cfg.n_steps_per_epoch)
        
        print("Finished!")

class TrainControllerRunner:
    
    def start(self, cfg):
        print("Running default runner")
        self.cfg = cfg

        self.datamodule = cfg.datamodule()
        self.modelmodule = cfg.modelmodule(
            dataset_info=self.datamodule.info,
        )
        trainer = cfg.trainer(
            callbacks=[v for k,v in cfg.callbacks.items()],
            logger=[v for k,v in cfg.logger.items()],
        )
        trainer.fit(
            model=self.modelmodule,
            datamodule=self.datamodule,
        )
        print("Finished!")

class TrainValuesRunner:
    
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
        
        # save diffuser training cfg for inference reload
        

        ### train

        n_epochs = int(cfg.global_cfg.n_train_steps // cfg.global_cfg.n_steps_per_epoch)
        for epoch in range(n_epochs):
            print(f'Epoch {epoch} / {n_epochs} | {cfg.output_dir}')
            trainer.train(n_train_steps=cfg.global_cfg.n_steps_per_epoch)
        
        print("Finished!")


def parse_diffusion(diffusion_dir, epoch, device, dataset_seed):
    """ parse diffusion model from 
    """
    diffusion_hydra_cfg_path = diffusion_dir + "/hydra_config.yaml"
    with open(diffusion_hydra_cfg_path, "r") as file:
        cfg = OmegaConf.load(file)
    cfg = hydra.utils.instantiate(cfg)
    ### init	
    dataset = cfg.dataset(seed=dataset_seed)
    render = cfg.render(dataset.env.name)
    
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    net = cfg.net(
        transition_dim=observation_dim + action_dim,
        cond_dim=observation_dim
    ).to(device)
    
    model = cfg.model(
        net,
        observation_dim=observation_dim,
        action_dim=action_dim,
    ).to(device)

    cfg.trainer.keywords['results_folder'] = diffusion_dir
    trainer = cfg.trainer(
        model,
        dataset,
        render,
    )
    # find latest
    import glob
    def get_latest_epoch(loadpath):
        states = glob.glob1(loadpath, 'state_*')
        latest_epoch = -1
        for state in states:
            epoch = int(state.replace('state_', '').replace('.pt', ''))
            latest_epoch = max(epoch, latest_epoch)
        return latest_epoch
    if epoch == 'latest':
        epoch = get_latest_epoch(diffusion_dir)
    print(f'\n[ parse diffusion model ] Loading model epoch: {epoch}\n')
    print(f'\n[ parse diffusion model ] Path: {diffusion_dir}\n')
    trainer.load(epoch)

    return trainer.ema_model, dataset, render

import numpy as np

class PlanGuidedRunner:
    CUSTOM_TARGET = {
        "tl2br": {
            "location": (1.0, 1.0),
            "target": np.array([7, 10]),
        },
        "br2tl": {
            "location": (7.0, 10.0),
            "target": np.array([1, 1]),
        },
        "tr2bl": {
            "location": (7.0, 1.0),
            "target": np.array([1, 10]),
        },
        "bl2tr": {
            "location": (1.0, 10.0),
            "target": np.array([7, 1]),
        },
        "2wayAv1": {
            "location": (1.0, 1.0),
            "target": np.array([3, 4]),
        },
        "2wayAv2": {
            "location": (3.0, 4.0),
            "target": np.array([1, 1]),
        },
        "2wayBv1": {
            "location": (5, 6),
            "target": np.array([1, 10]),
        },
        "2wayBv2": {
            "location": (1, 10.0),
            "target": np.array([5, 6]),
        },
    }
    def start(self, cfg):
        self.cfg = cfg

        diffuser = self.load_diffuser(cfg.diffuser.dir, cfg.diffuser.epoch)
        diffusion, dataset, self.renderer = diffuser.net.diffusion, diffuser.dynamic_cfg["dataset"], diffuser.dynamic_cfg["dataset"].renderer
        
        # diffusion, dataset, self.renderer = parse_diffusion(cfg.diffusion.dir, cfg.diffusion.epoch, cfg.device, cfg.diffusion.dataset_seed)

        guide = cfg.guide

        policy = cfg.policy(
            guide=guide,
            diffusion_model=diffusion,
            normalizer=dataset.normalizer,
        )

        policy.diffusion_model.to(cfg.device)

        env = dataset.env
        if "maze" not in env.name:
            assert cfg.trainer.custom_target is None, "Only maze environments need targets, so the cfg.trainer.custom_target should be None"
            assert cfg.trainer.use_controller_act is False, "Only maze environments can use controller, so the cfg.trainer.use_controller_act should be False"
        
        if cfg.trainer.custom_target is not None:
            # env.set_state(self.CUSTOM_TARGET[cfg.trainer.custom_target]["state"])
            env.set_target(self.CUSTOM_TARGET[cfg.trainer.custom_target]["target"])
            observation = env.reset_to_location(self.CUSTOM_TARGET[cfg.trainer.custom_target]["location"])
            print("###")
            print(f"use custom target ### {cfg.trainer.custom_target} ###")
            print("state", env.state_vector())
            print("target", env._target)
            print()
        else:
            observation = env.reset()

        ## observations for rendering
        rollout = [observation.copy()]
        
        total_reward = 0
        for t in range(cfg.trainer.max_episode_length):
            wandb_logs = {}
            if t % 10 == 0: print(cfg.output_dir, flush=True)

            ## save state for rendering only
            state = env.state_vector().copy()
            
            ## make action
            conditions = {0: observation}
            if "maze" in env.name: 
                conditions[diffusion.horizon-1] = np.array(list(env._target) + [0, 0])
            if t == 0: 
                actions, samples = policy(conditions, batch_size=cfg.trainer.batch_size, verbose=cfg.trainer.verbose)
                act_env = samples.actions[0][0] # (a_dim) only select one for act_
                sequence = samples.observations[0] # (horizon, s_dim)
                first_step_plan = samples.observations # (B, horizon, s_dim)
                first_step_conditions = conditions

                # import matplotlib.pyplot as plt
                # import torch
                # path="./debug/trace.png"
                # plt.figure()
                # x_ = samples.observations[:,:,[6,14,15]]
                # turn to cpu numpy
                # if isinstance(x_, torch.Tensor):
                #     if x_.is_cuda:
                #         x_ = x_.cpu()
                #     x_ = x_.detach().numpy()
                # coord_0 = x_[0]
                # T = coord_0.shape[0]
                # for dim in range(coord_0.shape[1]):
                # 	plt.plot(np.arange(T), coord_0[:, dim])
                # plt.savefig(path)
            else:
                if not cfg.trainer.plan_once:
                    actions, samples = policy(conditions, batch_size=cfg.trainer.batch_size, verbose=cfg.trainer.verbose)
                    act_env = samples.actions[0][0]
                    sequence = samples.observations[0]
            if cfg.trainer.use_controller_act:
                if t == diffusion.horizon - 1: 
                    next_waypoint = sequence[-1].copy() if cfg.trainer.plan_once else sequence[1]
                    next_waypoint[2:] = 0
                else:
                    next_waypoint = sequence[t+1] if cfg.trainer.plan_once else sequence[1]
                act_env = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
            
            ## execute action in environment
            next_observation, reward, terminal, _ = env.step(act_env)
 
            ## print reward and score
            total_reward += reward
            score = env.get_normalized_score(total_reward)
            print(
                f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
                f'values: {samples.values[0].item()}',
                flush=True,
            )
            wandb_logs["rollout/reward"] = reward
            wandb_logs["rollout/total_reward"] = total_reward
            wandb_logs["rollout/values"] = score
            guide_specific_metrics = guide.metrics(samples.observations)
            for key, value in guide_specific_metrics.items():
                wandb_logs[f"rollout/guide_{key}"] = value[0].item()

            ## update rollout observations
            rollout.append(next_observation.copy())

            ## render every `cfg.trainer.vis_freq` steps
            # self.log(t, samples, state, rollout, first_step_conditions)

            ## wandb log
            wandb_commit = (not cfg.wandb.lazy_commits_freq) or (t % cfg.wandb.lazy_commits_freq == 0)
            wandb.log(wandb_logs, step=t, commit=wandb_commit)

            ## end
            if terminal:
                break
            if cfg.trainer.plan_once and t == diffusion.horizon - 1:
                break
            
            observation = next_observation
        
        ### final log
        import os # TODO move
        wandb_logs = {}
        img_rollout_sample = self.renderer.render_rollout(
            os.path.join(self.cfg.output_dir, f'rollout_final.png'),
            rollout,
            first_step_conditions,
            fps=80,
        )
        wandb_logs["final/first_step_plan"] = wandb.Image(
            self.renderer.episodes2img(first_step_plan[:4])
        )
        wandb_logs["final/rollout"] = wandb.Image(img_rollout_sample)
        wandb_logs["final/total_reward"] = total_reward
        guide_specific_metrics = guide.metrics(np.stack(rollout)[None,:])
        for key, value in guide_specific_metrics.items():
            wandb_logs[f"final/guide_{key}"] = value[0].item()
        wandb.log(wandb_logs)

    def log(self, t, samples, state, rollout=None, conditions=None):
        import os
        wandb_logs = {}

        if t % self.cfg.vis_freq != 0:
            return

        ## render image of plans
        img_sample = self.renderer.composite(
            os.path.join(self.cfg.output_dir, f'{t}.png'),
            samples.observations[:4],
            conditions
        )
        # wandb_logs["samples"] = [wandb.Image(img_) for img_ in img_samples[0]]
        wandb_logs["samples"] = [wandb.Image(img_sample)]

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
    
    def load_diffuser(self, dir_, epoch_):
        print("\n\n\n### loading diffuser ...")
        from src.modelmodule import DiffuserModule
        diffuser_cfg = OmegaConf.load(Path(dir_)/"hydra_config.yaml")
        datamodule = hydra.utils.instantiate(diffuser_cfg.datamodule)()
        modelmodule = DiffuserModule.load_from_checkpoint(
            Path(dir_)/"checkpoints"/f"{epoch_}.ckpt",
            dataset_info=datamodule.info,
        )
        return modelmodule