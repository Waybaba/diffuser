import minari
import gymnasium as gym
import numpy as np

dataset_name = "pointmaze-umaze-v0"
total_steps = 10_000

# continuing task => the episode doesn't terminate or truncate when reaching a goal
# it will generate a new target. For this reason we set the maximum episode steps to
# the desired size of our Minari dataset (evade truncation due to time limit)
env = gym.make("PointMaze_Medium-v3", continuing_task=True, max_episode_steps=total_steps)

# Data collector wrapper to save temporary data while stepping. Characteristics:
#   * Custom StepDataCallback to add extra state information to 'infos' and divide dataset in different episodes by overridng
#     truncation value to True when target is reached
#   * Record the 'info' value of every step
collector_env = minari.DataCollectorV0(
    env, record_infos=True
)

obs, _ = collector_env.reset(seed=123)

for n_step in range(int(total_steps)):
    action = collector_env.env.action_space.sample()
    # Add some noise to each step action
    action += np.random.randn(*action.shape) * 0.5
    action = np.clip(
        action, env.action_space.low, env.action_space.high, dtype=np.float32
    )
    obs, rew, terminated, truncated, info = collector_env.step(action)
    print(n_step, rew, terminated, truncated, info)

dataset = minari.create_dataset_from_collector_env(
    collector_env=collector_env,
    dataset_id=dataset_name,
    algorithm_name="QIteration",
    code_permalink="https://github.com/Farama-Foundation/Minari/blob/main/docs/tutorials/dataset_creation/point_maze_dataset.py",
    author="Rodrigo Perez-Vicente",
    author_email="rperezvicente@farama.org",
)