import gymnasium as gym
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data


class StateIOEnv(gym.Env):
    """
    A simplified version of a State.io-style strategy game.
    The map consists of discrete nodes (bases) that can be controlled by players.
    The player sends units from one base to another to capture territory.
    """

    def __init__(self, renderflag = True, num_nodes = 5, seed=42):
        super().__init__()
        self.seed = seed
        np.random.seed(seed)
        
        self.enemy_enabled = False
        self.num_nodes = num_nodes         # Number of bases on the map
        self.speed = 3
        self.step_count = 0
        self.max_timestep = 1000
        
        self.renderflag = renderflag

        positions = {i: np.random.rand(2) * 100 for i in range(self.num_nodes)}  # 0 to 100 2d coordinates
        self.positions = positions
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


    # def encode_transfers(self, max_transfers=50):
    #     """
    #     Encode my_troop_transferring into a fixed-size numpy array.
    #     Each transfer is [src, dst, units, time_remaining]
    #     Output shape: (max_transfers * 4,)
    #     """
    #     transfers = []

    #     for (src, dst), troop_list in self.my_troop_transferring.items():
    #         for troop in troop_list:
    #             transfers.append([src, dst, troop["units"], troop["time_remaining"]])

    #     # Pad with zeros if not enough transfers
    #     while len(transfers) < max_transfers:
    #         transfers.append([0, 0, 0, 0])

    #     # Clip if too many transfers (or raise warning)
    #     transfers = transfers[:max_transfers]

    #     return np.array(transfers, dtype=np.int32).flatten()
    # def _construct_observation(self):
    #     """
    #     Combine troop distributions and encoded transfers into a flat observation vector.
    #     Final shape: (2 * num_nodes + 80,)
    #     """
    #     base_obs = np.concatenate([
    #         self.my_troop_distribution,          # shape = (num_nodes,)
    #         self.neutral_troop_distribution      # shape = (num_nodes,)
    #     ])
    #     transfer_obs = self.encode_transfers(max_transfers=20).flatten()  # shape = (max_transfers*4, )

    #     full_obs = np.concatenate([base_obs, transfer_obs])     # shape = (2*num_nodes + 80,)
    #     return full_obs.astype(np.int32)
    def _construct_observation(self) -> Data:
        """
        Constructs a PyG-compatible Data object with troop transfers encoded into edge attributes.
        """
        # 1. Node features: [my_troops, neutral_troops]
        x = torch.tensor([
            [self.my_troop_distribution[i], self.neutral_troop_distribution[i], self.positions[i][0], self.positions[i][1]]
            for i in range(self.num_nodes)
        ], dtype=torch.float)

        # 2. Edge index and rich edge attributes
        edge_index = []
        edge_attr = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j and self.G.has_edge(i, j):
                    distance = self.distance_matrix[i, j]

                    # Default values
                    total_units = 0.0
                    avg_time = 0.0
                    num_transfers = 0

                    if (i, j) in self.my_troop_transferring:
                        troop_list = self.my_troop_transferring[(i, j)]
                        num_transfers = len(troop_list)
                        total_units = sum(t["units"] for t in troop_list)
                        avg_time = sum(t["time_remaining"] for t in troop_list) / num_transfers if num_transfers > 0 else 0.0

                    edge_index.append([i, j])
                    edge_attr.append([
                        distance,
                        total_units,
                        avg_time,
                        num_transfers
                    ])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()   # [2, num_edges]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)                     # [num_edges, 4]

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def reset(self):
        """
        Resets the environment to its initial state.
        """
        super().reset(seed=self.seed)
        
        self.step_count = 0
        
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

        return state, info

    def step(self, action):
        """
        Process an action taken by the agent.
        Action is a tuple: (source_index, target_index, units_to_send)
        """
        src, dst= action
        self.step_count+=1
        done = False
        truncated = False
        truncated = (self.step_count>=self.max_timestep)
        
        reward = 0

        # Check valid action
        if (
            0 <= src < self.num_nodes and
            0 <= dst < self.num_nodes and
            src != dst
        ):
            # print(f"Action taken: Move from {src} to {dst}")
            # Deduct units from source base
            units = self.my_troop_distribution[src]

            if units>0:
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
                    # The case which capturing a base from neutral
                    neutral_units = self.neutral_troop_distribution[dst]
                    if total_arrival_units >= neutral_units:
                        # If arrived troop number > neutral troops we can capture it
                        self.my_troop_distribution[dst] = total_arrival_units - neutral_units
                        # Give an reward based on the number of elimination
                        reward += neutral_units * 0.5
                        self.neutral_troop_distribution[dst] = 0
                        reward += 5  # bonus for capturing a base
                    else:
                        # Even not occupying a base we will give rewards also
                        # reward += total_arrival_units * 0.2
                        self.neutral_troop_distribution[dst] -= total_arrival_units
                else:
                    # This is purely reinforcement between already occupying bases
                    self.my_troop_distribution[dst] += total_arrival_units
                    
            if still_moving:
                self.my_troop_transferring[(src, dst)] = still_moving
            else:
                del self.my_troop_transferring[(src, dst)]
                    
        # Panalty for taking times
        reward -= 0.1
        
        # Check termination (for now: all neutral captured)
        if np.all(self.neutral_troop_distribution == 0):
            done = True
            reward += 50  # bonus for finishing game

        # All my occupied nodes will produce troops as time increases
        mask = self.neutral_troop_distribution == 0
        self.my_troop_distribution[mask] += 1
        
        # Construct the current observation which includes troop distribution and all transferrings
        next_obs = self._construct_observation()
        info = {
            'neutral_troop_distribution': self.neutral_troop_distribution.copy(),
            'my_troop_distribution': self.my_troop_distribution.copy(),
            'my_troop_transferring': self.my_troop_transferring.copy()
        }
        # If set renderflag to true it will use matplotlib to do static render
        if self.renderflag:
            self.render()
        return next_obs, reward, done, truncated, info

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
            
        except Exception as e:
            print(f"Invalid input: {e}, using detault action waiting action")
            src, dst = -100, -100
        obs, reward, done, _, _ = env.step((src, dst))
        print("Observation:", obs)
        print("Reward:", reward)
        if done:
            print("Game Over!")
            break    

    print("Initial Observation:", obs)
    print("Initial Info:", _)
    env.close()