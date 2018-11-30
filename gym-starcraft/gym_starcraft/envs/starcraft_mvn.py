'''
Example implementation of M vs N units gym environment on top of base environment.
To use a custom reward structure or unit command generator for this class,
you can extend it and override any functions
'''
import numpy as np

from gym import spaces

import torchcraft.Constants as tcc
import gym_starcraft.utils as utils
import gym_starcraft.envs.starcraft_base_env as sc

# NOTE: Initial coordinates are to be given in exact x and y pixels
# Further on in every transaction we have cell as our unit from torchcraft
# So, distance factor must be considered during initialization
DISTANCE_FACTOR = 8


# M units vs N units, starcraft environment
class StarCraftMvN(sc.StarCraftBaseEnv):
    TIMESTEP_PENALTY = -0.01

    def __init__(self, args, final_init=True):
        self.args = args

        self.move_steps = ((0, 1), (1, 0), (0, -1), (-1, 0), (0, 0),
                           (1, 1), (1, -1),(-1, -1),(-1, 1))

        self.initialize_together = args.initialize_together
        self.initialize_enemy_together = args.initialize_enemy_together
        self.init_range_start = args.init_range_start
        self.init_range_end = args.init_range_end

        super(StarCraftMvN, self).__init__(final_init=final_init, **vars(args))

        # NOTE: We don't really need to do this, as it is done by kwargs init already in base
        self.nagents = args.nagents
        self.nenemies = args.nenemies

        if not final_init:
            return

        # TODO: Get this dynamically to support heterogenuous
        self.vision = tcc.staticvalues['sightRange'][self.my_unit_pairs[0][0]] / DISTANCE_FACTOR
        self.full_vision = args.full_vision
        self.free_movement = args.free_movement
        self.step_size = args.step_size

        if hasattr(args, 'unlimited_attack_range'):
            self.unlimited_attack_range = True
        else:
            self.unlimited_attack_range = False

        self.prev_actions = np.zeros(self.nagents)

    def _set_units(self):
        # First element is our unit's id, 1 is quantity,
        # -1, -1, init_range_start, init_range_end
        # say that randomly initialize x and y coordinates
        # within init_range_start and init_range_end
        self.my_unit_pairs = [(self.args.our_unit_type, 1, -1, -1,
                               self.init_range_start, self.init_range_end)
                              for _ in range(self.nagents)]

        self.enemy_unit_pairs = [(self.args.enemy_unit_type, 1, -1, -1,
                                  self.init_range_start, self.init_range_end)
                                 for _ in range(self.nenemies)]

        if self.initialize_together:
            # 0 for marine, 37 for zergling, 2 for vulture, 65 for zealot
            self.my_unit_pairs = [(self.args.our_unit_type, self.nagents, -1, -1,
                                   self.init_range_start, self.init_range_end)]

        if self.initialize_enemy_together:
            self.enemy_unit_pairs = [(self.args.enemy_unit_type, self.nenemies, -1, -1,
                                      self.init_range_start, self.init_range_end)]

    def _action_space(self):
        # Move up, down, left, right, stay, attack agents i to n
        self.nactions = len(self.move_steps) + self.nenemies

        # return spaces.Box(np.array(action_low), np.array(action_high))
        return spaces.MultiDiscrete([self.nactions])

    def _observation_space(self):
        # absolute x, absolute y, my_hp, my_cooldown, prev_action, (relative_x, relative_y, in_vision, enemy_hp, enemy_cooldown) * nenemy
        obs_low = [0.0, 0.0, 0.0, 0.0, 0.0] + [-1.0, -1.0, 0.0, 0.0, 0.0] * self.nenemies
        obs_high = [1.0, 1.0, 1.0, 1.0, 1.0] + [1.0, 1.0, 1.0, 1.0, 1.0] * self.nenemies

        return spaces.Box(np.array(obs_low), np.array(obs_high), dtype=np.float32)

    def _make_commands(self, actions):
        cmds = []
        if self.state1 is None or actions is None:
            return cmds

        # Hack for case when map is not purely cleaned for frame
        if len(self.state1.units[self.state1.player_id]) > self.nagents:
            return cmds

        enemy_unit = None

        for idx in range(self.nagents):
            agent_id = self.agent_ids[idx]

            if agent_id not in self.my_current_units:
                # Agent is probably dead
                continue

            my_unit = self.my_current_units[agent_id]
            action = actions[idx]
            prev_action = self.prev_actions[idx]

            if action < len(self.move_steps):
                new_x = my_unit.x + self.move_steps[action][0] * self.step_size
                new_y = my_unit.y + self.move_steps[action][1] * self.step_size

                new_x = min(new_x, self.init_range_end)
                new_y = min(new_y, self.init_range_end)

                new_x = max(new_x, self.init_range_start)
                new_y = max(new_y, self.init_range_start)

                # Move commands always override previous commands (required for kiting)
                cmds.append([
                    tcc.command_unit, my_unit.id,
                    tcc.unitcommandtypes.Move, -1, int(new_x), int(new_y), -1
                ])
            else:
                enemy_id = action - len(self.move_steps)

                enemy_id = self.enemy_ids[enemy_id]

                if enemy_id in self.enemy_current_units:
                    enemy_unit = self.enemy_current_units[enemy_id]
                else:
                    enemy_unit = None

                if not enemy_unit:
                    continue

                distance = utils.get_distance(my_unit.x, -my_unit.y,
                                              enemy_unit.x, -enemy_unit.y)

                unit_command = tcc.command_unit_protected

                # Send protected command only if previous command was attack
                if prev_action < len(self.move_steps):
                    unit_command = tcc.command_unit

                range_attribute = self.unit_attributes[my_unit.type]['rangeAttribute']

                # Should be in attack range to attack
                if distance <= getattr(my_unit, range_attribute) or self.unlimited_attack_range:
                    cmds.append([
                        unit_command, my_unit.id,
                        tcc.unitcommandtypes.Attack_Unit, enemy_unit.id
                    ])
        self.prev_actions = actions
        return cmds

    def _make_observation(self):
        myself = None
        enemy = None

        full_obs = np.zeros((self.nagents,) + self.observation_space.shape)

        for idx in range(self.nagents):
            agent_id = self.agent_ids[idx]

            if agent_id in self.my_current_units:
                myself = self.my_current_units[agent_id]
            else:
                myself = None

            if myself is None:
                continue

            curr_obs = full_obs[idx]
            curr_obs[0] = myself.x / self.state1.map_size[0]
            curr_obs[1] = myself.y / self.state1.map_size[1]

            # To simplify add unit's health and shield points
            curr_obs[2] = (myself.health + myself.shield) / (myself.max_health + myself.max_shield)

            cd = getattr(myself, self.unit_attributes[myself.type]['cdAttribute'])

            curr_obs[3] = cd / self.unit_attributes[myself.type]['maxCD']
            curr_obs[4] = self.prev_actions[idx] / self.nactions

            # Get observation for each enemy for each agent
            for enemy_idx in range(self.nenemies):
                enemy_id = self.enemy_ids[enemy_idx]
                if enemy_id in self.enemy_current_units:
                    enemy = self.enemy_current_units[enemy_id]
                else:
                    enemy = None

                if enemy is None:
                    continue

                if (myself.attacking or myself.starting_attack) and \
                    self.prev_actions[idx] == enemy_idx + len(self.move_steps):
                    self.attack_map[idx][enemy_idx] = 1

                distance = utils.get_distance(myself.x, myself.y, enemy.x, enemy.y)

                obs_idx = 5 + enemy_idx * 5

                if distance <= self.vision or self.full_vision:
                    curr_obs[obs_idx] = (myself.x - enemy.x) / (self.vision)
                    curr_obs[obs_idx + 1] = (myself.y - enemy.y) / (self.vision)
                    curr_obs[obs_idx + 2] = 0
                else:
                    curr_obs[obs_idx] = 0
                    curr_obs[obs_idx + 1] = 0
                    curr_obs[obs_idx + 2] = 1

                curr_obs[obs_idx + 3] = (enemy.health + enemy.shield) / (enemy.max_health + enemy.max_shield)
                cd = getattr(enemy, self.unit_attributes[enemy.type]['cdAttribute'])
                curr_obs[obs_idx + 4] =  cd / self.unit_attributes[enemy.type]['maxCD']

        return full_obs

    def _compute_reward(self):
        reward = np.zeros(self.nagents)

        for idx in range(self.nagents):
            if self.agent_ids[idx] in self.my_current_units:
                reward[idx] += self.TIMESTEP_PENALTY
            # Give own health difference as negative reward
            reward[idx] += self.obs[idx][2] - self.obs_pre[idx][2]

            for enemy_idx in range(self.nenemies):
                obs_idx = 5 + enemy_idx * 5
                # If the agent has attacked this enemy, then give diff in enemy's health as +ve reward
                if self.attack_map[idx][enemy_idx] == 1:
                    reward[idx] += self.obs_pre[idx][obs_idx + 3] - self.obs[idx][obs_idx + 3]

        return reward

    def reward_terminal(self):
        # Terminal reward based on whether we won or not
        reward = np.zeros(self.nagents)

        for idx in range(self.nagents):
            # Give terminal negative reward of each enemies' health
            for enemy_idx in range(self.nenemies):
                obs_idx = 5 + enemy_idx * 5
                # 3 is the best scaling factor we found in our tests
                reward[idx] += 0 - self.obs_pre[idx][obs_idx + 3] * 3

            # If the agent has attacked and we have won, give positive reward
            # which include some scaling factor of number of enemies and remaining health
            if self._has_won() == 1 and self.attack_map[idx].any():
                reward[idx] += +5 * self.nenemies + self.obs_pre[idx][2] * 3
            elif self.nagents == self.nenemies and len(self.my_current_units) > len(self.enemy_current_units):
                # Give some reward in case we didn't won but we have more units alive than enemy
                # Remove this to ensure agents have a destructive nature
                reward[idx] += 2
            else:
                # If it has finished, give whole agent's own health as negative reward
                reward[idx] += 0 - self.obs_pre[idx][2] * 3

        if self._has_won() == 1:
            self.episode_wins += 1

        return reward

    def _has_step_completed(self):
        return True

    def _get_info(self):
        # Add alive mask to info for use by downstream trainer
        alive_mask = np.ones(self.nagents)

        for idx in range(self.nagents):
            agent_id = self.agent_ids[idx]

            if agent_id not in self.my_current_units:
                alive_mask[idx] = 0

        info = {'alive_mask': alive_mask}
        info.update(super()._get_info())

        return info

    def step(self, action):
        return self._step(action)

    def reset(self):
        # Reset the environment for next step
        self.attack_map = np.zeros((self.nagents, self.nenemies))
        return self._reset()
