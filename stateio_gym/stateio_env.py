import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

class StateIOEnv(gym.Env):
    """
    A simplified version of a State.io-style strategy game.
    The map consists of discrete nodes (bases) that can be controlled by players.
    The player sends units from one base to another to capture territory.
    """

    def __init__(self):
        super().__init__()
        self.enemy_enabled = False
        self.num_nodes = 10          # Number of bases on the map
        self.max_units = 100         # Max number of units a base can hold
        self.speed = 5.0

    def encode_transfers(self, max_transfers=20):
        """
        Encode my_troop_transferring into a fixed-size numpy array.
        Each transfer is [src, dst, units, time_remaining]
        Output shape: (max_transfers * 4,)
        """
        transfers = []

        for (src, dst), troop_list in self.my_troop_transferring.items():
            for troop in troop_list:
                transfers.append([src, dst, troop["units"], troop["time_remaining"]])

        # Pad with zeros if not enough transfers
        while len(transfers) < max_transfers:
            transfers.append([0, 0, 0, 0])

        # Clip if too many transfers (or raise warning)
        transfers = transfers[:max_transfers]

        return np.array(transfers, dtype=np.int32).flatten()
    def _construct_observation(self):
        """
        Combine troop distributions and encoded transfers into a flat observation vector.
        Final shape: (2 * num_nodes + 80,)
        """
        base_obs = np.concatenate([
            self.my_troop_distribution,          # shape = (num_nodes,)
            self.neutral_troop_distribution      # shape = (num_nodes,)
        ])
        transfer_obs = self.encode_transfers(max_transfers=20).flatten()  # shape = (max_transfers*4, )

        full_obs = np.concatenate([base_obs, transfer_obs])     # shape = (2*num_nodes + 80,)
        return full_obs.astype(np.int32)
    
    def reset(self, seed=42):
        """
        Resets the environment to its initial state.
        """
        super().reset(seed=seed)
        # Create a random graph to represent the map
        positions = {i: np.random.rand(2) * 100 for i in range(self.num_nodes)}  # 0~100范围内的2D坐标
        G = nx.Graph()
        for i in range(self.num_nodes):
            G.add_node(i, pos=positions[i])

        # Add edges with Euclidean distances as weights
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                dist = np.linalg.norm(positions[i] - positions[j])  # 欧氏距离
                G.add_edge(i, j, weight=dist)

        # Visualize the graph
        pos = nx.get_node_attributes(G, 'pos')
        edge_labels = nx.get_edge_attributes(G, 'weight')

        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.1f}" for k, v in edge_labels.items()}, font_size=8)
        plt.title("Graph with Euclidean Edge Weights")
        plt.show()

        self.neutral_troop_distribution = np.ones(self.num_nodes, dtype=np.int32) * 20  # Neutral bases start with 20 units each
        self.my_troop_distribution = np.zeros(self.num_nodes, dtype=np.int32)  # Player starts with no units
        self.my_starting_base = 0  # Player's starting base index

        self.my_troop_distribution[self.my_starting_base] = 50  # Player starts with 50 units at their base
        # Action space: [source_base_index, target_base_index]
        self.neutral_troop_distribution[self.my_starting_base] = 0  # Player's base is no longer neutral
        
        self.my_troop_transferring = {} 
        # Things like {
        #     (0, 4): [ {"units": 10, "time_remaining": 3} ],
        #     (1, 7): [ {"units": 5, "time_remaining": 1}, {"units": 6, "time_remaining": 2} ]
        # }
        state = self._construct_observation()
        info = {
            'neutral_troop_distribution': self.neutral_troop_distribution,
            'my_troop_distribution': self.my_troop_distribution,
            'my_troop_transferring': self.my_troop_transferring
        }

        return state.flatten(), info

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

if __name__ == "__main__":
    # Example usage
    env = StateIOEnv()
    obs, _ = env.reset()
    # env.render()

    # for _ in range(10):
    #     action = env.action_space.sample()
    #     obs, reward, done, truncated, info = env.step(action)
    #     print(f"Action: {action}, Reward: {reward}, Done: {done}")
    #     env.render()
    #     if done:
    #         break
    print("Initial Observation:", obs)
    print("Initial Info:", _)
    env.close()