import numpy as np
import gym_starcraft.envs.starcraft_explore as sc

class StarCraftExploreComm(sc.StarCraftExplore):
    def __init__(self, args, final_init=True):
        if not args.enemy_comm:
            raise RuntimeError('Explore mode comm can only be run with enemy comm')
        args.nagents -= 1
        super(StarCraftExploreComm, self).__init__(args, final_init)
        args.nagents += 1
        self.nfriendly = args.nfriendly

    def _make_observation(self):
        full_obs = np.zeros((self.nfriendly + 1, ) + self.observation_space.shape)

        full_obs[:self.nfriendly] = super()._make_observation()

        enemy_id = self.enemy_ids[0]

        if enemy_id not in self.enemy_current_units:
            return full_obs

        enemy_unit = self.enemy_current_units[enemy_id]

        curr_obs = full_obs[self.nfriendly]

        curr_obs[0] = enemy_unit.x / self.state1.map_size[0]
        curr_obs[1] = enemy_unit.y / self.state1.map_size[1]
        curr_obs[2] = 0

        return full_obs

    def _compute_reward(self):
        reward = np.zeros(self.nfriendly + 1)

        reward[:self.nfriendly] = super()._compute_reward()

        on_prey = np.count_nonzero(self.near_enemy)

        if on_prey == 0:
            reward[self.nfriendly] -= self.TIMESTEP_PENALTY
        else:
            reward[self.nfriendly] = 0
        return reward

    def reward_terminal(self):
        super()._compute_reward()
        return np.zeros(self.nfriendly + 1)

    def _get_info(self):
        return {}
