import gym


env = gym.make('BipedalWalker-v3')
ep_rounds = 2
for ep in range(ep_rounds):
    obs = env.reset()
    for t in range(100):
        env.render()
        print(obs)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            print("{} timesteps taken for this ep {}.".format(t+1, ep))
            break
env.close()
