from gymnasium.envs.registration import register

register(
    id="StateIO-v0",  
    entry_point="stateio_gym.stateio_env:StateIOEnv", 
)