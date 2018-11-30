import gym


# Useful only for our project as a wrapper to load StarCraft envs
class StarCraftWrapperEnv(gym.Env):

    def __init__(self):

        self.__version__ = '0.0.1'


    def init_args(self, parser):
        env = parser.add_argument_group('Starcraft tasks')
        env.add_argument('--task_type', type=str, default='mvn',
                         help='Type of starcraft task')
        env.add_argument('--nenemies', type=int, default=1,
                         help='Number of enemies')
        env.add_argument('--torchcraft_dir', type=str, default='~/TorchCraft',
                         help='TorchCraft directory')
        env.add_argument('--bwapi_launcher_path', type=str,
                         default='../bwapi/bin/BWAPILauncher',
                         help='Path to BWAPILauncher')
        env.add_argument('--config_path', type=str,
                         default='../gym-starcraft/gym_starcraft/envs/config.yml',
                         help='Path to TorchCraft/OpenBW yml config')
        env.add_argument('--server_ip', type=str, default='127.0.0.1',
                         help='IP of the server')
        env.add_argument('--server_port', type=int, default=11111,
                         help='Port of the server')
        env.add_argument('--ai_type', type=str, default='builtin',
                         help='Type of AI, builtin|attack_closest|attack_weakest')
        env.add_argument('--speed', type=int, default=0,
                         help='Speed')
        env.add_argument('--init_range_start', type=int, default=100,
                         help='Start of initialization range of units')
        env.add_argument('--init_range_end', type=int, default=150,
                         help='End of initialization range of units')
        env.add_argument('--frame_skip', type=int, default=1,
                         help='Frame skip')
        env.add_argument('--our_unit_type', type=int, default=0,
                         help="Our unit type (0: Marine, 37: Zergling)")
        env.add_argument('--enemy_unit_type', type=int, default=0,
                         help="Enemy unit type (0: Marine, 37: Zergling)")
        env.add_argument('--set_gui', action="store_true", default=False,
                         help="Show GUI")
        env.add_argument('--initialize_together', action="store_true", default=False,
                            help="Initialize our units together")
        env.add_argument('--initialize_enemy_together', action="store_true", default=False,
                            help="Initialize enemy's units together")
        env.add_argument('--self_play', action='store_true', default=False,
                         help='Should play with self')
        env.add_argument('--full_vision', action='store_true', default=False,
                         help='Full vision on map')
        env.add_argument('--free_movement', action='store_true', default=False,
                         help='Free movement on map')
        env.add_argument('--enemy_comm', action='store_true', default=False,
                         help='Test for enemy communication')
        env.add_argument('--step_size', type=int, default=8,
                         help="Step of the agent, Default: 8")


        # Explore args
        env.add_argument('--stay_near_enemy', action='store_true', default=False,
                         help='Once enemy found stay near it')
        env.add_argument('--cooperation_setting', type=str, default='normal',
                         help="Cooperations setting in explore mode " +
                         "(normal|cooperative|competitive)")
        env.add_argument('--explore_vision', type=int, default=10,
                         help="Vision of the agent, Default: 10")

    def multi_agent_init(self, args, final_init):
        # TODO: Later with more task_types we will use switch args.task_type:
        if args.task_type == 'explore':
            from gym_starcraft.envs.starcraft_explore import StarCraftExplore
            Class = StarCraftExplore
        elif args.task_type == 'explore_comm':
            from gym_starcraft.envs.starcraft_explore_comm import StarCraftExploreComm
            Class = StarCraftExploreComm
        else:
            from gym_starcraft.envs.starcraft_mvn import StarCraftMvN
            Class = StarCraftMvN

        self.env = Class(args, final_init)
