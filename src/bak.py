import gymnasium as gym
# import gym

try:
  print('Checking that the installation succeeded:')
  import mujoco
  mujoco.MjModel.from_xml_string('<mujoco/>')
except Exception as e:
  raise e from RuntimeError(
      'Something went wrong during installation. Check the shell output above '
      'for more information.\n'
      'If using a hosted Colab runtime, make sure you enable GPU acceleration '
      'by going to the Runtime menu and selecting "Choose runtime type".')

print('Installation successful.')

# 创建环境
# env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave', 'kettle'], render_mode="rgb_array")
# AdroitHandHammer-v1
env = gym.make('AdroitHandHammer-v1', render_mode="rgb_array")

# 环境初始化
observation = env.reset()

# 显示初始观察
print('Initial observation:', observation)

done = False
while not done:
    # 随机选择一个动作
    action = env.action_space.sample()
    
    # 执行动作并观察结果
    observation, reward, done, _, info = env.step(action)
    imgs = env.render() # "rgb_array
    # 显示结果
    # print('Action:', action)
    # print('Observation:', observation)
    # print('Reward:', reward)
    # print('Done:', done)
    # print('Info:', info)
    print(imgs.shape)

# 关闭环境
env.close()
