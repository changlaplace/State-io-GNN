# main.py
from stateio_env import StateIOEnv
from pygameui import pygame_loop

if __name__ == "__main__":
    env = StateIOEnv(renderflag=False)
    env.reset()
    pygame_loop(env)


