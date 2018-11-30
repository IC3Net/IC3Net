import gym

import torchcraft.Constants as tcc
import random
import yaml
import subprocess
import sys
import time
import socket, errno
import os
import signal
import atexit
import uuid
import gym_starcraft.utils as utils
import tempfile


DISTANCE_FACTOR = 8
class StarCraftBaseEnv(gym.Env):
    def __init__(self, torchcraft_dir='~/TorchCraft',
                 config_path='./config.yml', **kwargs):
        """Initialized the base StarCraft class which initializes TorchCraft server
        connects to it and handles environment steps and resets.

        Keyword Arguments:
            torchcraft_dir {str} -- Directory of TorchCraft repository
            (TorchCraft should be installed in that directory) (default: {'~/TorchCraft'})
            config_path {str} -- Path for configuration yml file (default: {'./config.yml'})
        """

        self.init_from_kwargs(kwargs)

        self.action_space = self._action_space()
        self.observation_space = self._observation_space()

        self.config_path = config_path
        self.torchcraft_dir = torchcraft_dir


        if not self.final_init:
            return

        options = self.load_config_options()

        self.start_torchcraft(options)

        self.episodes = 0
        self.episode_wins = 0
        self.episode_steps = 0
        self.first_reset = True
        self._set_unit_attributes()

        # NOTE: These should be overrided in derived class
        # Should be a list of pairs where each pair is
        # (quantity, unit_type, x, y, start_coordinate, end_coordinate)
        # So (1, 0, -1, -1, 100, 150) will instantiate 0 type unit
        # at a random place between x = (100, 150) and y = (100, 150)
        # Leave empty if you want to instantiate anywhere in whole map
        self.vision = 3
        self.my_unit_pairs = []
        self.enemy_unit_pairs = []
        self.my_current_units = {}
        self.enemy_current_units = {}
        self.agent_ids = []
        self.enemy_ids = []
        self.state1 = None
        self.obs = None
        self.obs_pre = None
        self.stat = {}
        self._set_units()

    def init_from_kwargs(self, kwargs):
        """Init the base keyword arguments passed and set then as class properties"""
        default_kw_args = {
            # IP of the server where TorchCraft will listen
            'server_ip': '127.0.0.1',
            # Speed passed to TorchCraft
            'speed': 0,
            # AI Type: builtin | attack_closest | attack_weakest,
            'ai_type': 'builtin',
            # Enable fog of war if false
            'full_vision': False,
            # Number of frame to skip
            'frame_skip': 1,
            # Pass to enable GUI
            'set_gui': 0,
            # Maximum number of episode steps
            'max_episode_steps': 200,
            # Pass this false when you don't want to start the server yet
            # but init the class for other purposes (e.g. getting action space)
            'final_init': True,
            # Print summary at end of episode
            'print_summary': False,
            # Number of our agents
            'nagents': 1,
            # Number of enemy agents
            'nenemies': 1
        }

        if kwargs is None:
            kwargs = {}
        default_kw_args.update(kwargs)
        kwargs = default_kw_args

        self.server_ip = kwargs['server_ip']
        self.frame_skip = kwargs['frame_skip']
        self.speed = kwargs['speed']
        self.max_episode_steps = kwargs['max_steps']
        self.set_gui = kwargs['set_gui']
        self.ai_type = kwargs['ai_type']
        self.full_vision = kwargs['full_vision']
        self.final_init = kwargs['final_init']
        self.print_summary = kwargs['print_summary']
        self.nagents = kwargs['nagents']
        self.nenemies = kwargs['nenemies']

    def load_config_options(self):
        """Load config options from config file and environment"""
        config = None
        with open(self.config_path, 'r') as f:
            try:
                config = yaml.load(f)
            except yaml.YAMLError as err:
                print('Config yaml error', err)
                sys.exit(0)

        self.bwapi_launcher_path = config['options']['BWAPI_INSTALL_PREFIX']

        # Check if environment contains BWAPI_INSTALL path
        if 'BWAPI_INSTALL_PREFIX' in os.environ:
            self.bwapi_launcher_path = os.environ['BWAPI_INSTALL_PREFIX']

        self.bwapi_launcher_path = os.path.join(self.bwapi_launcher_path,
                                                'bin', 'BWAPILauncher')

        # Set environment variables to be passed directly to BWAPI
        tmpfile = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        options = dict(os.environ)
        for key, val in config['options'].items():
            options[key] = str(val)


        options['BWAPI_CONFIG_AUTO_MENU__GAME_TYPE'] = "USE MAP SETTINGS"
        options['BWAPI_CONFIG_AUTO_MENU__AUTO_RESTART'] = "ON"
        # Use LAN and Local mode to start a self-play kind of mode
        options['BWAPI_CONFIG_AUTO_MENU__AUTO_MENU'] = "LAN"
        options['OPENBW_LAN_MODE'] = "LOCAL"
        options['OPENBW_LOCAL_PATH'] = tmpfile
        options['BWAPI_CONFIG_AUTO_MENU__MAP'] = os.path.abspath(options['BWAPI_CONFIG_AUTO_MENU__MAP'])

        return options

    def start_torchcraft(self, options):
        """Starts torchcraft on available port and given IP with passed options"""
        cmds = []
        cmds.append(os.path.expanduser(self.bwapi_launcher_path))

        proc1 = subprocess.Popen(cmds,
                                 cwd=os.path.expanduser(self.torchcraft_dir),
                                 env=options,
                                 stdout=subprocess.PIPE
                )
        self._register_kill_at_exit(proc1)

        proc2 = subprocess.Popen(cmds,
                                 cwd=os.path.expanduser(self.torchcraft_dir),
                                 env=options,
                                 stdout=subprocess.PIPE
                )
        self._register_kill_at_exit(proc2)

        matchstr = b"TorchCraft server listening on port "
        for line in iter(proc1.stdout.readline, ''):
            if len(line) != 0:
                print(line.rstrip().decode('utf-8'))
            if line[:len(matchstr)] == matchstr:
                self.server_port1 = int(line[len(matchstr):].strip())
                break

        for line in iter(proc2.stdout.readline, ''):
            if len(line) != 0:
                print(line.rstrip().decode('utf-8'))
            if line[:len(matchstr)] == matchstr:
                self.server_port2 = int(line[len(matchstr):].strip())
                break

    def init_conn(self):
        """Init connection with torchcraft server"""
        # Import torchcraft in this function so that torchcraft is not an explicit
        # dependency for projects importing this repo
        import torchcraft as tc
        self.client1 = tc.Client()
        self.client1.connect(self.server_ip, self.server_port1)
        self.state1 = self.client1.init()

        self.client2 = tc.Client()
        self.client2.connect(self.server_ip, self.server_port2)
        self.state2 = self.client2.init()

        setup = [[tcc.set_combine_frames, 1],
                 [tcc.set_speed, self.speed],
                 [tcc.set_gui, self.set_gui],
                 # NOTE: We use custom frameskip method now
                 # Skip frame below
                 [tcc.set_frameskip, 1],
                 [tcc.set_cmd_optim, 1]]

        self.client1.send(setup)
        self.state1 = self.client1.recv()
        self.client2.send(setup)
        self.state2 = self.client2.recv()

    def __del__(self):
        if hasattr(self, 'client') and self.client1:
            self.client1.close()

    def _register_kill_at_exit(self, proc):
        atexit.register(proc.kill)

    def _kill_child(self, child_pid):
        if child_pid is None:
            pass
        else:
            os.kill(child_pid, signal.SIGTERM)

    def _set_unit_attributes(self):
        # Creating a map for easy access of max cooldowns and other things
        # NOTE: At the moment, this wrapper supports only air vs air and ground vs ground
        # matches, some minor changes will be required to support mix.
        self.unit_attributes = {
            # Marine
            0: {
                'cdAttribute': 'groundCD',
                'maxCD': 15,
                'rangeAttribute': 'groundRange'
            },
            # Vulture
            2: {
                'cdAttribute': 'groundCD',
                'maxCD': 30,
                'rangeAttribute': 'groundRange'
            },
            # Wraith
            8: {
                'cdAttribute': 'airCD',
                'maxCD': 22,
                'rangeAttribute': 'airRange'
            },
            # Corsair
            60: {
                'cdAttribute': 'airCD',
                'maxCD': 8,
                'rangeAttribute': 'airRange'
            },
            # Zealot
            65: {
                'cdAttribute': 'groundCD',
                'maxCD': 22,
                'rangeAttribute': 'groundRange'
            },
            # Zergling
            37: {
                'cdAttribute': 'groundCD',
                'maxCD': 8,
                'rangeAttribute': 'groundRange'
            },
            # Mutalisk
            43: {
                'cdAttribute': 'airCD',
                'maxCD': 30,
                'rangeAttribute': 'airRange'
            },
            # Medic
            34: {
                'cdAttribute': 'groundCD',
                'maxCD': 1,
                'rangeAttribute': 'groundRange'
            }
        }

    def _step(self, action):
        """Given an action, do an environment step.
        This makes commands for TorchCraft, sends them and gets back the reward
        Also update statistics and returns new observation based on the action taken
        """
        # Stop stepping if map config has come into play
        if len(self.state1.aliveUnits.values()) > self.nagents + self.nenemies:
            reward = self._compute_reward()
            self.my_current_units = {}
            self.obs = self._make_observation()
            done = True
            info = {}
            return self.obs, reward, done, info

        self.episode_steps += 1

        cmds = self._make_commands(action)
        self.client1.send(cmds)
        self.state1 = self.client1.recv()

        enemy_cmds = self._get_enemy_commands()

        self.client2.send(enemy_cmds)
        self.state2 = self.client2.recv()

        self._skip_frames()

        while not self._has_step_completed():
            self._skip_frames(1)

        self.obs = self._make_observation()
        reward = self._compute_reward()
        done = self._check_done()
        info = self._get_info()

        self._update_stat()
        self.obs_pre = self.obs
        return self.obs, reward, done, info

    def _empty_step(self):
        """Make an empty step where we don't send anything to server"""
        self.client1.send([])
        self.state1 = self.client1.recv()
        self.client2.send([])
        self.state2 = self.client2.recv()

    def _skip_frames(self, skips=-1):
        if skips == -1:
            skips = self.frame_skip

        count = 0

        while count < skips:
            self._empty_step()
            count += 1

    def _get_enemy_commands(self):
        """Get enemy commands based on the 'ai_type'
        NOTE: Override this function in case you want custom enemy AI.
        TODO: Initialize func for AI only once.
        """
        cmds = []
        func = lambda *args: None

        if self.ai_type == 'attack_closest':
            func = utils.get_closest
        elif self.ai_type == 'attack_weakest':
            func = utils.get_weakest

        for unit in self.state2.units[self.state2.player_id]:
            opp_unit = func(unit, self.state2, self.state1.player_id)

            if opp_unit is None:
                continue
            dist = utils.get_distance(opp_unit.x, opp_unit.y, unit.x, unit.y)
            vision = tcc.staticvalues['sightRange'][unit.type] / DISTANCE_FACTOR

            # Check if the our unit is in range of enemy then attack
            if (dist <= vision or self.full_vision):
                cmds.append([
                    tcc.command_unit_protected, unit.id,
                    tcc.unitcommandtypes.Attack_Unit, opp_unit.id
                ])

        # No-op or empty cmds means in built AI will be emulated
        return cmds

    def try_killing(self):
        """Keeps sending commands to server for killing units
        until they don't wipe off the map"""

        if not self.state1:
            return

        while len(self.state1.units[self.state1.player_id]) != 0 \
              or len(self.state2.units[self.state2.player_id]) != 0:
            c1units = self.state1.units[self.state1.player_id]
            c2units = self.state2.units[self.state2.player_id]

            self.client1.send(self.kill_units(c1units))
            self.state1 = self.client1.recv()

            self.client2.send(self.kill_units(c2units))
            self.state2 = self.client2.recv()

            for _ in range(10):
                self._empty_step()

    def _reset(self):
        """Reset after episode end for next episode"""
        wins = self.episode_wins
        episodes = self.episodes

        if self.print_summary:
            print("Episodes: %4d | Wins: %4d | WinRate: %1.3f" % (
                    episodes, wins, wins / (episodes + 1E-6)))

        self.episodes += 1
        self.episode_steps = 0

        if self.first_reset:
            self.init_conn()
            self.first_reset = False

        # Try killing active units
        self.try_killing()

        c1 = []
        c2 = []

        # Create the units for new episode
        for unit_pair in self.my_unit_pairs:
            c1 += self._get_create_units_command(self.state1.player_id, unit_pair)

        for unit_pair in self.enemy_unit_pairs:
            c2 += self._get_create_units_command(self.state2.player_id, unit_pair)

        # Send commands to both clients
        self.client1.send(c1)
        self.state1 = self.client1.recv()
        self.client2.send(c2)
        self.state2 = self.client2.recv()

        # Wait for units to appear on the map
        while len(self.state1.units.get(self.state1.player_id, [])) == 0 \
              and len(self.state2.units.get(self.state2.player_id, [])) == 0:
            self._empty_step()

        # This adds my_units and enemy_units to object.
        self.my_current_units = self._parse_units_to_unit_dict(self.state1.units[self.state1.player_id])
        self.enemy_current_units = self._parse_units_to_unit_dict(self.state2.units[self.state2.player_id])

        # This adds my and enemy's units' ids as incrementing list
        self.agent_ids = list(self.my_current_units)
        self.enemy_ids = list(self.enemy_current_units)
        self.stat = {}

        # Create the observation for current step
        self.obs = self._make_observation()
        self.obs_pre = self.obs

        return self.obs

    def _get_create_units_command(self, player_id, unit_pair):
        """Generates command for creating units"""

        defaults = [1, 100, 100, 0, self.state1.map_size[0] - 10][len(unit_pair) - 1:]
        unit_type, quantity, x, y, start, end = (list(unit_pair) + defaults)[:6]

        return self.create_units(player_id, quantity, x=x, y=y,
                                 unit_type=unit_type, start=start,
                                 end=end)

    def create_units(self, player_id, quantity, unit_type=0, x=100, y=100, start=0, end=250):
        """Create units in specific location (x, y) within bounding box of
        (start, start) and (end, end). If either of x or y is -1, that coordindate is
        initialized randomly within the range specified.

        Arguments:
            player_id {int} -- ID of the player for which to create units
            quantity {int} -- Number of units to create

        Keyword Arguments:
            unit_type {int} -- ID of unit type to create, 0 is marine (default: {0})
            x {int} -- X coordinate of initialization. If -1, it is random (default: {100})
            y {int} -- Y coordinate of initialization. If -1, it is random (default: {100})
            start {int} -- Start of bounding box (default: {0})
            end {int} -- End of bounding box. (default: {250})

        Default arguments of start and end cover whole map. If you see some errors regarding
        division with zero when you default "end" of 250, try decreasing it.
        """

        # NOTE: If you want some custom kind of initialization, override this function and
        # call parent's create_units in that function
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

    def is_empty(self, data):
        return data is not None and len(data) == 0

    def kill_units(self, units):
        """Tries to kill argument units by passing commands to BWAPI
        [units] is a list of units to be killed.
        """
        commands = []

        for u in units:
            command = [
                tcc.command_openbw,
                tcc.openbwcommandtypes.KillUnit,
                u.id
            ]
            commands.append(command)
        return commands


    def _parse_units_to_unit_dict(self, units, units_type='my_units'):
        """Convert units to a dict for easy usage"""
        unit_dict = dict()

        for unit in units:
            unit_dict[unit.id] = unit

        return unit_dict

    def _action_space(self):
        """Returns a space object"""
        raise NotImplementedError

    def _observation_space(self):
        """Returns a space object"""
        raise NotImplementedError

    def _make_commands(self, action):
        """Returns a game command list based on the action"""
        raise NotImplementedError

    def _make_observation(self):
        """Returns a observation object based on the game state"""
        raise NotImplementedError

    def _has_step_completed(self):
        """Returns a boolean to tell whether the current step has
        actually completed in the game"""
        raise NotImplementedError

    def _compute_reward(self):
        """Returns a computed scalar value based on the game state"""
        raise NotImplementedError

    def _set_units(self):
        """Sets my units as per specification mentioned in init
        Override this method in derived class and just pass if you
        don't intend to initialize any units
        """
        raise NotImplementedError

    def _check_done(self):
        """Returns true if the episode was ended"""
        # If either of my units or enemy units has a count of 0 or if
        # we have reached max steps then we are finished
        return (len(self.state1.units[self.state1.player_id]) == 0 or \
                len(self.state2.units[self.state2.player_id]) == 0 or \
                self.episode_steps == self.max_episode_steps)

    def _has_won(self):
        # Our units should be more than 0 and enemy units should be 0
        return (
            len(self.state1.units[self.state1.player_id]) > 0 and \
            len(self.state2.units[self.state2.player_id]) == 0
        )

    def _get_info(self):
        """Returns a dictionary contains debug info"""
        return {
            'state1': self.state1,
            'state2': self.state2
        }

    def _update_stat(self):
        if self._check_done():
            if self._has_won():
                self.stat['success'] = 1
            else:
                self.stat['success'] = 0

            self.stat['steps_taken'] = self.episode_steps

        return self.stat

    def render(self, mode='human', close=False):
        """Implement in case you want to render some specific information
        """
        raise NotImplementedError
