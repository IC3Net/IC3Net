import argparse
def get_parser():
    parser = argparse.ArgumentParser('Starcraft agent')
    parser.add_argument('--nagents', type=int, default=1,
                        help='Number of enemies')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Max steps')
    parser.add_argument('--nenemies', type=int, default=1,
                        help='Number of enemies')
    parser.add_argument('--torchcraft_dir', type=str, default='~/TorchCraft',
                        help='TorchCraft directory')
    parser.add_argument('--bwapi_launcher_path', type=str,
                        default='../bwapi/bin/BWAPILauncher',
                        help='Path to BWAPILauncher')
    parser.add_argument('--config_path', type=str,
                        default='../gym-starcraft/gym_starcraft/envs/config.yml',
                        help='Path to TorchCraft/OpenBW yml config')
    parser.add_argument('--server_ip', type=str, default='127.0.0.1',
                        help='IP of the server')
    parser.add_argument('--server_port', type=int, default=11111,
                        help='Port of the server')
    parser.add_argument('--ai_type', type=str, default='builtin',
                        help='Type of AI, builtin|attack_closest|attack_weakest')
    parser.add_argument('--speed', type=int, default=0,
                        help='Speed')
    parser.add_argument('--init_range_start', type=int, default=0,
                        help='Start of initialization range of units')
    parser.add_argument('--init_range_end', type=int, default=250,
                        help='End of initialization range of units')
    parser.add_argument('--frame_skip', type=int, default=1,
                        help='Frame skip')
    parser.add_argument('--our_unit_type', type=int, default=0,
                        help="Our unit type (0: Marine, 37: Zergling)")
    parser.add_argument('--enemy_unit_type', type=int, default=0,
                        help="Enemy unit type (0: Marine, 37: Zergling)")
    parser.add_argument('--set_gui', action="store_true", default=False,
                        help="Show GUI")
    parser.add_argument('--initialize_together', action="store_true", default=False,
                        help="Initialize our units together")
    parser.add_argument('--initialize_enemy_together', action="store_true", default=False,
                        help="Initialize enemy's units together")
    parser.add_argument('--self_play', action='store_true', default=False,
                        help='Should play with self')
    parser.add_argument('--full_vision', action='store_true', default=False,
                        help='Unlimited vision on map')
    parser.add_argument('--free_movement', action='store_true', default=False,
                        help='Unlimited movement on map')
    parser.add_argument('--unlimited_attack_range', action='store_true', default=False,
                        help='Attack range over full map')
    parser.add_argument('--enemy_comm', action='store_true', default=False,
                        help='Test for enemy communication')
    parser.add_argument('--step_size', type=int, default=8,
                        help="Step of the agent, Default: 8")
    return parser