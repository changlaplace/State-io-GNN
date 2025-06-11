import os
import torch
from stateio_gym.stateio_env import StateIOEnv
from gnn_policy import GNNPolicy
from ppo_agent import select_action, compute_returns
import pygame
import numpy as np
import torch.nn.functional as F
import time

def pygame_loop(env, policy):
    X_SCALE_RATIO = 7
    Y_SCALE_RATIO = 5.5

    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    def draw_text(text, pos, color=(0, 0, 0)):
        surface = font.render(text, True, color)
        screen.blit(surface, pos)

    obs, _ = env.reset()
    done, truncated = False, False
    reward = None
    running = True

    policy.eval()  # inference mode

    while running:
        screen.fill((255, 255, 255))

        if not (done or truncated):
            with torch.no_grad():
                if ADDRANDOM:
                    action, log_prob, entropy, edge_id = select_action(policy, obs)
                    obs, reward, done, truncated, _ = env.step(action)
                else:
                    logits = policy(obs)
                    probs = F.softmax(logits, dim=0)
                    action_index = torch.argmax(probs).item()  # greedy
                    src = obs.edge_index[0, action_index].item()
                    dst = obs.edge_index[1, action_index].item()
                    obs, reward, done, truncated, info = env.step((src, dst))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if done:
            print(f"You have won the game in {env.step_count} steps!")
        elif truncated:
            print(f"Game truncated after {env.step_count} steps.")
        
        # ---- 3. Draw map ----
        draw_text(f"Step: {env.step_count}/{env.max_timestep}, Reward: {reward}", (40, 40), (0, 0, 0))

        for (i, j) in env.G.edges():
            x1, y1 = int(env.positions[i][0]*X_SCALE_RATIO), int(env.positions[i][1]*Y_SCALE_RATIO)
            x2, y2 = int(env.positions[j][0]*X_SCALE_RATIO), int(env.positions[j][1]*Y_SCALE_RATIO)
            pygame.draw.line(screen, (200, 200, 200), (x1, y1), (x2, y2), 2)

        for i, pos in env.positions.items():
            x, y = int(pos[0]*X_SCALE_RATIO), int(pos[1]*Y_SCALE_RATIO)
            color = (0, 255, 0) if env.neutral_troop_distribution[i] > 0 else (255, 100, 100)
            pygame.draw.circle(screen, color, (x, y), 20)
            draw_text(f"M:{env.my_troop_distribution[i]}", (x - 18, y - 30), (0, 0, 255))
            draw_text(f"N:{env.neutral_troop_distribution[i]}", (x - 18, y + 25), (0, 128, 0))

        for (src, dst), troop_list in env.my_troop_transferring.items():
            x0, y0 = int(env.positions[src][0]*X_SCALE_RATIO), int(env.positions[src][1]*Y_SCALE_RATIO)
            x1, y1 = int(env.positions[dst][0]*X_SCALE_RATIO), int(env.positions[dst][1]*Y_SCALE_RATIO)
            for troop in troop_list:
                t_total = env.distance_matrix[src, dst] / env.speed
                frac = 1 - troop["time_remaining"] / t_total
                frac = np.clip(frac, 0.0, 1.0)
                xt = int(x0 + frac * (x1 - x0))
                yt = int(y0 + frac * (y1 - y0))
                pygame.draw.circle(screen, (0, 0, 0), (xt, yt), 5)
                draw_text(str(troop["units"]), (xt - 10, yt - 20))
        
        pygame.display.flip()
        
        if done or truncated:
            time.sleep(5)
            break
        clock.tick(30)  

    pygame.quit()
  

if __name__ == "__main__":
    IFLOAD = True

    ADDRANDOM = True
    model_folder = r'./models'
    model_name = r'transfer_5to10_20250610_174512.pt'

    policy = GNNPolicy(in_channels=4, edge_feat_dim=4, hidden_dim=64, use_attention=True)
    if IFLOAD:
        policy.load_state_dict(torch.load(os.path.join(model_folder, model_name)))
    policy.eval()

    env = StateIOEnv(renderflag=False, num_nodes=100, seed=42)
    pygame_loop(env, policy)