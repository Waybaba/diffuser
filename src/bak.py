import gymnasium as gym
# import gym

# 创建环境
env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave', 'kettle'], render_mode="rgb_array")

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
    print('Action:', action)
    print('Observation:', observation)
    print('Reward:', reward)
    print('Done:', done)
    print('Info:', info)

# 关闭环境
env.close()
