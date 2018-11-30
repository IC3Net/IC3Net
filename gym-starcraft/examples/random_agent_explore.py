from gym_starcraft.envs.starcraft_explore import StarCraftExplore
from flags import get_parser


class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()

if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--stay_near_enemy', action='store_true', default=False,
                        help='Once enemy found stay near it')
    parser.add_argument('--cooperation_setting', type=str, default='normal',
                        help="Cooperations setting in explore mode " +
                        "(normal|cooperative|competitive)")
    parser.add_argument('--explore_vision', type=int, default=10,
                        help="Vision of the agent, Default: 10")
    args = parser.parse_args()
    env = StarCraftExplore(args, final_init=True)

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
