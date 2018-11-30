from gym_starcraft.envs.starcraft_mvn import StarCraftMvN
from flags import get_parser


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()

if __name__ == '__main__':
    args = get_parser().parse_args()
    env = StarCraftMvN(args, final_init=True)

    agent = RandomAgent(env.action_space)
    episodes = 0

    while episodes < 50:
        obs = env.reset()
        done = False
        while not done:
            actions = []

            for _ in range(args.nagents):
                action = agent.act()[0]
                actions.append(action)
            obs, reward, done, info = env.step(actions)
        episodes += 1
        print(reward)

    env.close()
