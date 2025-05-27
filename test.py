import gymnasium as gym
import stateio_gym  # required to trigger registration

env = gym.make("StateIO-v0")
obs, _ = env.reset()
env.render()

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")
    env.render()
    if done:
        break

env.close()