from gym.envs.registration import register

register(
    id='Starcraft-MvN-v0',
    entry_point='gym_starcraft.envs.starcraft_mvn:StarCraftMvN',
)

register(
    id='StarCraftWrapper-v0',
    entry_point='gym_starcraft.envs.starcraft_wrapper_env:StarCraftWrapperEnv',
    # kwargs={'nb_agents' : 1},
)
