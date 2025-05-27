import gymnasium as gym
from gymnasium import spaces
import numpy as np

class StateIOEnv(gym.Env):
    """
    A simplified version of a State.io-style strategy game.
    The map consists of discrete nodes (bases) that can be controlled by players.
    The player sends units from one base to another to capture territory.
    """

    def __init__(self):
        super().__init__()

        self.num_nodes = 10          # Number of bases on the map
        self.max_units = 100         # Max number of units a base can hold

        # Action space: [source_base_index, target_base_index]
        self.action_space = spaces.MultiDiscrete([self.num_nodes, self.num_nodes])

        # Observation space: for each node, two values [ownership, unit_count]
        # ownership: 0 = neutral, 1 = player, 2 = enemy
        self.observation_space = spaces.Box(
            low=np.array([[0, 0]] * self.num_nodes).flatten(),
            high=np.array([[2, self.max_units]] * self.num_nodes).flatten(),
            dtype=np.int32
        )

        self.state = None

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        """
        super().reset(seed=seed)

        # Initialize all nodes to neutral with random unit counts
        self.state = np.zeros((self.num_nodes, 2), dtype=np.int32)
        self.state[:, 1] = np.random.randint(10, 30, size=self.num_nodes)

        # Set ownership of first node to player, last node to enemy
        self.state[0, 0] = 1
        self.state[-1, 0] = 2

        return self.state.flatten(), {}

    def step(self, action):
        """
        Executes an action in the environment.
        Args:
            action: [source_node, target_node]
        Returns:
            observation, reward, terminated, truncated, info
        """
        src, dst = action
        reward = 0
        done = False
        truncated = False
        info = {}

        # Invalid actions (same node or not player's base)
        if src == dst or self.state[src, 0] != 1:
            reward = -1
            return self.state.flatten(), reward, done, truncated, info

        attack_force = self.state[src, 1] // 2
        if attack_force <= 0:
            reward = -0.5
            return self.state.flatten(), reward, done, truncated, info

        self.state[src, 1] -= attack_force

        dst_owner = self.state[dst, 0]
        dst_units = self.state[dst, 1]

        if attack_force > dst_units:
            # Successful capture
            self.state[dst, 0] = 1
            self.state[dst, 1] = attack_force - dst_units
            reward = 1.0
        else:
            # Failed or partial attack
            self.state[dst, 1] -= attack_force
            reward = -0.1

        # Win condition: all enemy nodes eliminated
        if np.all(self.state[:, 0] != 2):
            done = True
            reward += 10

        return self.state.flatten(), reward, done, truncated, info

    def render(self):
        """
        Print the current state of the map.
        """
        print("==== MAP STATE ====")
        for i, (owner, units) in enumerate(self.state):
            owner_str = ["Neutral", "Player", "Enemy"][owner]
            print(f"Node {i:02d}: {owner_str} | Units: {units}")
        print("===================")

    def close(self):
        """
        Optional resource cleanup.
        """
        pass
