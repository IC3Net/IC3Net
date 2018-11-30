import numpy as np
from gym import spaces

import torchcraft.Constants as tcc
import gym_starcraft.utils as utils
import gym_starcraft.envs.starcraft_mvn as sc
import random

DISTANCE_FACTOR = 8

class StarCraftExplore(sc.StarCraftMvN):
    TIMESTEP_PENALTY = -0.05
    ONPREY_REWARD = 0.05

    def __init__(self, args, final_init=True):
        if args.nenemies != 1:
            raise RuntimeError('Only 1 enemy allowed in this case')

        if args.enemy_unit_type != 34 or args.our_unit_type != 34:
            print("Warning: Only medic can be used as unit in explore mode")

        args.enemy_unit_type = 34
        args.our_unit_type = 34

        super(StarCraftExplore, self).__init__(args, final_init)

        if not hasattr(args, 'cooperation_setting'):
            args.cooperation_setting = 'normal'

        if not hasattr(args, 'explore_vision'):
            args.explore_vision = 10

        if not hasattr(args, 'stay_near_enemy'):
            args.stay_near_enemy = False

        if args.cooperation_setting == 'normal':
            self.prey_exponent = 0
            self.ONPREY_REWARD = 0
        elif args.cooperation_setting == 'cooperative':
            self.prey_exponent = 1
        else:
            self.prey_exponent = -1

        self.vision = args.explore_vision
        self.near_enemy = np.zeros(self.nagents)
        self.stay_near_enemy = args.stay_near_enemy
        self.step_size = args.step_size


    def _action_space(self):
        self.nactions = len(self.move_steps)

        return spaces.MultiDiscrete([self.nactions])

    def _observation_space(self):
        # absolute x, absolute y, (relative_x, relative_y, in_vision) * nenemy
        obs_low = [0.0, 0.0] + [-1.0, -1.0, 0.0] * self.nenemies
        obs_high = [1.0, 1.0] + [1.0, 1.0, 1.0] * self.nenemies

        return spaces.Box(np.array(obs_low), np.array(obs_high), dtype=np.float32)

    def _has_step_completed(self):
        return True


    def _make_commands(self, actions):
        cmds = []

        if self.state1 is None or actions is None:
            return cmds

        for idx in range(self.nagents):
            agent_id = self.agent_ids[idx]

            if agent_id not in self.my_current_units:
                continue

            my_unit = self.my_current_units[agent_id]
            action = actions[idx]

            if self.near_enemy[idx] == 1 and self.stay_near_enemy:
                cmds.append([
                    tcc.command_unit,
                    my_unit.id,
                    tcc.unitcommandtypes.Stop
                ])
                continue

            if action >= len(self.move_steps):
                cmds.append([
                    tcc.command_unit,
                    my_unit.id,
                    tcc.unitcommandtypes.Stop
                ])
                continue

            new_x = my_unit.x + self.move_steps[action][0] * self.step_size
            new_y = my_unit.y + self.move_steps[action][1] * self.step_size

            new_x = min(new_x, self.init_range_end)
            new_y = min(new_y, self.init_range_end)

            new_x = max(new_x, self.init_range_start)
            new_y = max(new_y, self.init_range_start)

            cmds.append([
                tcc.command_unit, my_unit.id,
                tcc.unitcommandtypes.Move, -1, int(new_x), int(new_y), -1
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

                obs_idx = 2 + enemy_idx * 3

                if distance <= self.vision or self.full_vision:
                    curr_obs[obs_idx] = (myself.x - enemy.x) / (self.vision)
                    curr_obs[obs_idx + 1] = (myself.y - enemy.y) / (self.vision)
                    curr_obs[obs_idx + 2] = 0
                else:
                    curr_obs[obs_idx] = 0
                    curr_obs[obs_idx + 1] = 0
                    curr_obs[obs_idx + 2] = 1
        return full_obs

    def _get_enemy_commands(self):
        return []


    def create_units(self, player_id, quantity, unit_type=0, x=100, y=100, start=0, end=256):
        if x < 0:
            x = (random.randint(0, (end - start)) + start) * DISTANCE_FACTOR

        if y < 0:
            y = (random.randint(0, (end - start)) + start) * DISTANCE_FACTOR
        commands = []

        for _ in range(quantity):
            command = [
                tcc.command_openbw,
                tcc.openbwcommandtypes.SpawnUnit,
                player_id,
                unit_type,
                x,
                y,
            ]
            commands.append(command)

        return commands


    def _compute_reward(self):
        reward = np.zeros(self.nagents)

        enemy_id = self.enemy_ids[0]
        enemy = self.enemy_current_units[enemy_id]

        for idx in range(self.nagents):
            if self.agent_ids[idx] not in self.my_current_units:
                continue

            unit = self.my_current_units[self.agent_ids[idx]]

            dist = utils.get_distance(unit.x, unit.y, enemy.x, enemy.y)

            if dist <= self.vision:
                self.near_enemy[idx] = 1
            else:
                self.near_enemy[idx] = 0

        for idx in range(self.nagents):
            if self.agent_ids[idx] not in self.my_current_units:
                continue

            if self.near_enemy[idx] == 1:
                reward[idx] += self.ONPREY_REWARD * (np.count_nonzero(self.near_enemy) ** self.prey_exponent)
            else:
                reward[idx] += self.TIMESTEP_PENALTY

        return reward

    def reward_terminal(self):
        reward = np.zeros(self.nagents)

        if self._has_won() == 1:
            self.episode_wins += 1

        return reward

    def _has_won(self):
        return np.count_nonzero(self.near_enemy) == self.nagents


    def _check_done(self):
        return (
            self.episode_steps == self.max_episode_steps or \
            (self.ONPREY_REWARD == 0 and np.count_nonzero(self.near_enemy) == self.nagents)
        )
