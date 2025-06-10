import gymnasium as gym
import stateio_gym  # required to trigger registration
from stateio_gym.stateio_env import StateIOEnv
import numpy as np
# env = gym.make("StateIO-v0")
# obs, _ = env.reset()
# env.render()

# for _ in range(10):
#     action = env.action_space.sample()
#     obs, reward, done, truncated, info = env.step(action)
#     print(f"Action: {action}, Reward: {reward}, Done: {done}")
#     env.render()
#     if done:
#         break

# env.close()


env1 = StateIOEnv(renderflag=False, num_nodes=5, seed=42)
env2 = StateIOEnv(renderflag=False, num_nodes=5, seed=42)
env1.reset()
env2.reset()

same_positions = all(np.allclose(env1.positions[i], env2.positions[i]) for i in range(env1.num_nodes))

same_troop = np.array_equal(env1.my_troop_distribution, env2.my_troop_distribution)

print("Same positions:", same_positions)
print("Same troop distribution:", same_troop)