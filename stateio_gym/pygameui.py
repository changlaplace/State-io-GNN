# ui.py
import pygame
import numpy as np

def pygame_loop(env):
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    selected_base = None

    def draw_text(text, pos, color=(0, 0, 0)):
        surface = font.render(text, True, color)
        screen.blit(surface, pos)

    running = True
    while running:
        screen.fill((255, 255, 255))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                for i, pos in env.positions.items():
                    px, py = int(pos[0]*7), int(pos[1]*5.5)
                    if (mx - px)**2 + (my - py)**2 < 20**2:
                        if selected_base is None:
                            selected_base = i
                        else:
                            env.step((selected_base, i))
                            selected_base = None

        # Draw edges
        for (i, j) in env.G.edges():
            x1, y1 = int(env.positions[i][0]*7), int(env.positions[i][1]*5.5)
            x2, y2 = int(env.positions[j][0]*7), int(env.positions[j][1]*5.5)
            pygame.draw.line(screen, (200, 200, 200), (x1, y1), (x2, y2), 2)

        # Draw nodes
        for i, pos in env.positions.items():
            x, y = int(pos[0]*7), int(pos[1]*5.5)
            color = (0, 255, 0) if env.neutral_troop_distribution[i] > 0 else (255, 100, 100)
            pygame.draw.circle(screen, color, (x, y), 20)
            draw_text(f"M:{env.my_troop_distribution[i]}", (x - 18, y - 30), (0, 0, 255))
            draw_text(f"N:{env.neutral_troop_distribution[i]}", (x - 18, y + 25), (0, 128, 0))
            if selected_base == i:
                pygame.draw.circle(screen, (0, 0, 0), (x, y), 24, 2)

        # Draw moving troops
        for (src, dst), troop_list in env.my_troop_transferring.items():
            x0, y0 = int(env.positions[src][0]*7), int(env.positions[src][1]*5.5)
            x1, y1 = int(env.positions[dst][0]*7), int(env.positions[dst][1]*5.5)
            for troop in troop_list:
                t_total = env.distance_matrix[src, dst] / env.speed
                frac = 1 - troop["time_remaining"] / t_total
                frac = np.clip(frac, 0.0, 1.0)
                xt = int(x0 + frac * (x1 - x0))
                yt = int(y0 + frac * (y1 - y0))
                pygame.draw.circle(screen, (0, 0, 0), (xt, yt), 5)
                draw_text(str(troop["units"]), (xt - 10, yt - 20))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
