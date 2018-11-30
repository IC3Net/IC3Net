import math

from gym_starcraft.envs.starcraft_mvn import StarCraftMvN
from gym_starcraft.utils import get_closest
from flags import get_parser

class AttackClosestAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def set_move_steps(self, num):
        self.move_actions = num

    def act(self, obs):
        min_value = math.inf
        min_unit = None

        enemy_id = 0
        for i in range(5, len(obs), 5):
            x = obs[i]
            y = obs[i + 1]

            if min_value > x * x + y * y:
                min_value = x * x + y * y
                min_unit = enemy_id
            enemy_id += 1

        return self.move_actions + min_unit

if __name__ == '__main__':
    args = get_parser().parse_args()
    args.unlimited_attack_range = True
    args.unlimited_vision = True

    env = StarCraftMvN(args, final_init=True)

    agent = AttackClosestAgent(env.action_space)
    agent.set_move_steps(len(env.move_steps))
    episodes = 0
    success = 0

    while episodes < 500:
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            actions = []

            for i in range(args.nagents):
                action = agent.act(obs[i])
                actions.append(action)
            obs, reward, done, info = env.step(actions)
            total_reward += reward

        total_reward += env.reward_terminal()
        success += env.stat['success']
        episodes += 1
        state1 = info['state1']
        state2 = info['state2']
        print("Reward: ", total_reward)
        print("Success: ", success / episodes)
        print("Alive:", "Mine:", len(state1.units[state1.player_id]),
              "Theirs:", len(state2.units[state2.player_id]))
    env.close()
