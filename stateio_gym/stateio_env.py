import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import time
from matplotlib import animation

class StateIOEnv(gym.Env):
    """
    A simplified version of a State.io-style strategy game.
    The map consists of discrete nodes (bases) that can be controlled by players.
    The player sends units from one base to another to capture territory.
    """

    def __init__(self, renderflag = True):
        super().__init__()
        
        self.enemy_enabled = False
        self.num_nodes = 5         # Number of bases on the map
        self.max_units = 10       # Max number of units a base can hold
        self.speed = 20
        self.renderflag = renderflag

        positions = {i: np.random.rand(2) * 100 for i in range(self.num_nodes)}  # 0~100范围内的2D坐标
        G = nx.Graph()
        for i in range(self.num_nodes):
            G.add_node(i, pos=positions[i])

        self.distance_matrix = np.zeros(shape=(self.num_nodes, self.num_nodes))
        self.G = G

        # Add edges with Euclidean distances as weights
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                dist = np.linalg.norm(positions[i] - positions[j])  # 
                self.distance_matrix[i,j]=dist
                self.distance_matrix[j,i]=dist
                G.add_edge(i, j, weight=dist)
        if self.renderflag:
            self.fig, self.ax = plt.subplots(figsize=(8, 6))


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
        if self.renderflag:
            self.render()

        return state.flatten(), info

    def step(self, action):
        """
        Process an action taken by the agent.
        Action is a tuple: (source_index, target_index, units_to_send)
        """
        src, dst= action

        done = False
        reward = 0

        # Check valid action
        if (
            0 <= src < self.num_nodes and
            0 <= dst < self.num_nodes and
            src != dst
        ):
            print(f"Action taken: Move from {src} to {dst}")
            # Deduct units from source base
            units = self.my_troop_distribution[src]

            self.my_troop_distribution[src] = 0
            
            time_to_arrive = self.distance_matrix[src, dst] / self.speed  # Compute time based on distance / speed
            
            
            # Add to transfers
            if (src, dst) not in self.my_troop_transferring:
                self.my_troop_transferring[(src, dst)] = []
            self.my_troop_transferring[(src, dst)].append({
                "units": units,
                "time_remaining": time_to_arrive
                })
            
        # Update transferring troops
        completed_transfers = []
        for (src, dst), troop_list in list(self.my_troop_transferring.items()):
            for troop in troop_list:
                troop["time_remaining"] -= 1
            # Extract completed ones
            arrivals = [t for t in troop_list if t["time_remaining"] <= 0]
            still_moving = [t for t in troop_list if t["time_remaining"] > 0]
            if arrivals:
                # Apply effect of arrival
                total_arrival_units = sum(t["units"] for t in arrivals)
                if self.neutral_troop_distribution[dst] > 0:
                    if total_arrival_units >= self.neutral_troop_distribution[dst]:
                        reward += 1  # captured a neutral base
                        self.my_troop_distribution[dst] = total_arrival_units - self.neutral_troop_distribution[dst]
                        self.neutral_troop_distribution[dst] = 0
                        reward += 5  # bonus for capturing a base
                    else:
                        self.neutral_troop_distribution[dst] -= total_arrival_units
                else:
                    self.my_troop_distribution[dst] += total_arrival_units
            if still_moving:
                self.my_troop_transferring[(src, dst)] = still_moving
            else:
                del self.my_troop_transferring[(src, dst)]

        # Check termination (for now: all neutral captured)
        if np.all(self.neutral_troop_distribution == 0):
            done = True
            reward += 100  # bonus for finishing game

        mask = self.neutral_troop_distribution == 0
        self.my_troop_distribution[mask] += 1
        next_obs = self._construct_observation()
        info = {
            'neutral_troop_distribution': self.neutral_troop_distribution.copy(),
            'my_troop_distribution': self.my_troop_distribution.copy(),
            'my_troop_transferring': self.my_troop_transferring.copy()
        }
        if self.renderflag:
            self.render()
        return next_obs, reward, done, False, info

    def render(self, rendermode = 'matplotlib'):
        """
        Render the current state of the environment.
        """
        if rendermode == 'matplotlib':
            G = self.G.copy()
            pos = nx.get_node_attributes(G, 'pos')
            edge_labels = nx.get_edge_attributes(G, 'weight')

            self.ax.clear()
            nx.draw(G, pos, ax=self.ax, with_labels=True, node_color='lightblue', node_size=500)
            nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.1f}" for k, v in edge_labels.items()}, font_size=8, ax=self.ax)
            self.ax.set_title("Graph with Euclidean Edge Weights")
            for node in G.nodes:
                x, y = pos[node]
                self.ax.text(x, y + 3, f"{self.my_troop_distribution[node]}", color='red', fontsize=10, ha='center')
                self.ax.text(x, y - 3, f"{self.neutral_troop_distribution[node]}", color='green', fontsize=10, ha='center')
            
            for (src, dst), troop_list in self.my_troop_transferring.items():
                for troop in troop_list:
                    time_remaining = troop["time_remaining"]
                    distance = self.distance_matrix[src, dst]
                    if distance == 0:
                        continue  # avoid div0

                    time_total = distance / self.speed
                    frac = 1 - time_remaining / time_total
                    frac = np.clip(frac, 0.0, 1.0)  # 防止越界

                    x0, y0 = pos[src]
                    x1, y1 = pos[dst]
                    x = x0 + frac * (x1 - x0)
                    y = y0 + frac * (y1 - y0)

                    self.ax.plot(x, y, 'ro', markersize=5)
                    self.ax.text(x, y + 1.5, f"{troop['units']}", color='black', fontsize=8, ha='center')
            plt.pause(0.1)
        elif rendermode == 'pygame':
            pass


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
    while True:
        print("Please input your action (source_index space target_index):")
        action_input = input()
        try:
            src, dst = map(int, action_input.split(' '))
            obs, reward, done, _, _ = env.step((src, dst))
            print("Observation:", obs)
            print("Reward:", reward)
            if done:
                print("Game Over!")
                break
        except Exception as e:
            print(f"Invalid input: {e}, using detault action waiting action")
            src, dst = None, None

    print("Initial Observation:", obs)
    print("Initial Info:", _)
    env.close()