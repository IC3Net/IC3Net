# gym-starcraft

This repository provides an OpenAI Gym interface to StarCraft: BroodWars online multiplayer game.

## Features

- Includes a base environment which can be easily extended to various specific environment and doesn't assume anything about the downstream task.
- Both enemy and our unit's commands can be overridden for scenarios in which either of units need to be controlled for deterministic settings.
- Includes an example derived class for `M vs N` environment which can be further extended for specific cases of general `M vs N` scenarios.
- In `M vs N` environment, any unit type with any quantity can be initialized anywhere on map or within a specific bounding box. This environment can also be used in case of StarCraft community-building tasks as buildings themselves are units.
- Includes an explore mode environment, where one can test exploration mode for agents in big and dynamic map of StarCraft.
- Attack closest and random agent included as an example agent implementation to be used with environment.
- MvN example supports partial observable setting in which vision is limited as in fog of war.
- Supports built-in, attack-closest and attack-weakest AI strategies.

### Combat Mode
![Combat Mode](https://i.imgur.com/sQGASF1.gif)

### Explore Mode
![Explore Mode](https://i.imgur.com/BywLKaH.gif)

## Prerequisites and Installation

- First, we need to install TorchCraft with OpenBW support. We have written down a [wiki](https://github.com/IC3Net/IC3Net/blob/master/gym-starcraft/wiki/installation.md) for this.
- Make sure to install python bindings for TorchCraft as well.
- Run `python setup.py develop` for installing this repo.

## Running

- First, update the [config](https://github.com/IC3Net/IC3Net/blob/master/gym-starcraft/gym_starcraft/envs/config.yml) present in `gym-starcraft/envs/config.yml` as per your requirements. Explanation is present in comments.
- Make sure all prerequisites are completed.
- To run a sample `attack_closest` match between 10 marines and 3 zealots in bounding box of (100, 100) to (150, 150) with GUI, run the following command:

```
 python examples/attack_closest.py --server_ip 127.0.0.1 --torchcraft_dir=TORCHCRAFT_DIR --set_gui --nagents 10 --max_steps 200 --frame_skip 8 --nenemies 3 --our_unit_type 0 --enemy_unit_type 65 --init_range_start 100 --init_range_end 150 --full_vision --unlimited_attack_range --initialize_enemy_together --step_size 16
```

Most of the other flags are self explanatory.

Use `python examples/attack_closest.py -h` for other options that are available.

## Custom Environment Development

- First, decide whether you can use either of combat MvN or explore mode environment as a start point to develop your custom environment. If you can do that, derive your new environment by extending one of these classes otherwise extend `StarCraftBaseEnv` like below:
```py
import gym_starcraft.envs.starcraft_base_env as sc

class YourCustomSCEnv(sc.StarCraftBaseEnv):
    def __init__(self, args):
        # Either use argparse namespace and pass as dict
        # or pass each of the arguments using specific keywords
        super(YourCustomSCEnv, self).__init__(**vars(args))
```

- Now you will need to update or implement some of the required functions. In case you are using either of MvN or explore mode, then you may skip some of these so they default to original implementation. Otherwise, you need to implement most of the required functions which we list below one by one. For each of the function, see sample implementation in MvN environment:

    - First, implement `_set_units` function in which you set `self.my_unit_pairs` and `self.enemy_unit_pairs` which are used to instantiate our and enemy units.
    - Second, implement `_action_space` and `_observation_space` if required.
    - Third, implement `_make_commands` function which takes `actions` as parameter and returns a list of commands in TorchCraft format. See sample implementation for an example.
    - Now, implement `_make_observation` function which return an numpy array of shape defined in `_observation_space` function.
    - `_has_step_completed` function is checked to make sure current step is completed in `_step` function by default. This can be implemented in case you need to make custom checks.
    - `_compute_reward` function must return reward for current step in case you are planning to use it. See `attack_closest` agent to see how reward is retrieved for each agent from environment.
    - `reward_terminal` function is used to calculate reward at the end of the episode and can be called by the trainer.
    - `step` function implemented as per gym specification must call internal `_step` at some point to calculate observation from StarCraft.
    - `reset` function implemented as per gym specification must call internal `_reset` at some point to reset the actual StarCraft environment through BWAPI and return initial observation.
    - Other functions include `_get_info` which returns info for current step and `_get_enemy_commands` which can be overriden to implement custom AI for StarCraft.

## Credits

Initial implementation of this package was based out of Alibaba's [gym-starcraft](https://github.com/alibaba/gym-starcraft) which didn't work properly with latest TorchCraft version.

## License

Code for this project is available under MIT license.


## TODO

- Support for heterogenuous agents requires some changes in vision range calculation (Vision range is calculated only once)
