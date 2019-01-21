# IC3Net

This repository contains reference implementation for IC3Net paper (accepted to ICLR 2019), **Learning when to communicate at scale in multiagent cooperative and competitive tasks**, available at [https://arxiv.org/abs/1812.09755](https://arxiv.org/abs/1812.09755)

## Cite

If you use this code or IC3Net in your work, please cite the following:

```
@article{singh2018learning,
  title={Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Tasks},
  author={Singh, Amanpreet and Jain, Tushar and Sukhbaatar, Sainbayar},
  journal={arXiv preprint arXiv:1812.09755},
  year={2018}
}
```

## Standalone environment version

- Find `gym-starcraft` at this repository: [apsdehal/gym-starcraft](https://github.com/apsdehal/gym-starcraft)
- Find `ic3net-envs` at this repository: [apsdehal/ic3net-envs](https://github.com/apsdehal/ic3net-envs)

## Installation

First, clone the repo and install ic3net-envs which contains implementation for Predator-Prey and Traffic-Junction

```
git clone https://github.com/IC3Net/IC3Net
cd IC3Net/ic3net-envs
python setup.py develop
```

**Optional**: If you want to run experiments on StarCraft, install the `gym-starcraft` package included in this package. Follow the instructions provided in README inside that packages.


Next, we need to install dependencies for IC3Net including PyTorch. For doing that run:

```
pip install -r requirements.txt
```

## Running

Once everything is installed, we can run the using these example commands

Note: We performed our experiments on `nprocesses` set to 16, you can change it according to your machine, but the plots may vary.

Note: Use `OMP_NUM_THREADS=1` to limit the number of threads spawned

### Predator-Prey

- IC3Net on easy version

```
python main.py --env_name predator_prey --nagents 3 --nprocesses 16 --num_epochs 2000 --hid_size 128 --detach_gap 10 --lrate 0.001 --dim 5 --max_steps 20 --ic3net --vision 0 --recurrent
```

- CommNet on easy version

```
python main.py --env_name predator_prey --nagents 3 --nprocesses 16 --num_epochs 2000 --hid_size 128 --detach_gap 10 --lrate 0.001 --dim 5 --max_steps 20 --commnet --vision 0 --recurrent
```

- IC on easy version

```
python main.py --env_name predator_prey --nagents 3 --nprocesses 16 --num_epochs 2000 --hid_size 128 --detach_gap 10 --lrate 0.001 --dim 5 --max_steps 20 --vision 0 --recurrent
```

- IRIC on easy version

```
python main.py --env_name predator_prey --nagents 3 --nprocesses 16 --num_epochs 2000 --hid_size 128 --detach_gap 10 --lrate 0.001 --dim 5 --max_steps 20 --mean_ratio 0 --vision 0 --recurrent
```

For medium version, change the following arguments:
- `nagents` to 5
- `max_steps` to 40
- `vision` to 1
- `dim` to 10


For hard version, change the following arguments:
- `nagents` to 10
- `max_steps` to 80
- `vision` to 1
- `dim` to 20


### Traffic Junction

- IC3Net on easy version

```
python main.py --env_name traffic_junction --nagents 5 --nprocesses 16 --num_epochs 2000 --hid_size 128 --detach_gap 10 --lrate 0.001 --dim 6 --max_steps 20 --ic3net --vision 0 --recurrent --add_rate_min 0.1 --add_rate_max 0.3 --curr_start 250 --curr_end 1250 --difficulty easy
```

- CommNet on easy version

```
python main.py --env_name predator_prey --nagents 5 --nprocesses 16 --num_epochs 2000 --hid_size 128 --detach_gap 10 --lrate 0.001 --dim 6 --max_steps 20 --commnet --vision 0 --recurrent  --add_rate_min 0.1 --add_rate_max 0.3 --curr_start 250 --curr_end 1250 --difficulty easy
```

- IC on easy version

```
python main.py --env_name predator_prey --nagents 5 --nprocesses 16 --num_epochs 2000 --hid_size 128 --detach_gap 10 --lrate 0.001 --dim 6 --max_steps 20 --vision 0 --recurrent  --add_rate_min 0.1 --add_rate_max 0.3 --curr_start 250 --curr_end 1250 --difficulty easy
```

- IRIC on easy version

```
python main.py --env_name predator_prey --nagents 5 --nprocesses 16 --num_epochs 2000 --hid_size 128 --detach_gap 10 --lrate 0.001 --dim 6 --max_steps 20 --mean_ratio 0 --vision 0 --recurrent --add_rate_min 0.1 --add_rate_max 0.3 --curr_start 250 --curr_end 1250 --difficulty easy
```

For medium version, change the following arguments:
- `nagents` to 10
- `max_steps` to 40
- `dim` to 14
- `add_rate_min` to 0.05
- `add_rate_max` to 0.02
- `difficulty` to medium


For hard version, change the following arguments:
- `nagents` to 20
- `max_steps` to 80
- `dim` to 18
- `add_rate_min` to 0.02
- `add_rate_max` to 0.05
- `difficulty` to hard

### StarCraft

Make sure you have gym-starcraft properly installed and configuration properly configured.

For explore task 50x50, 10Medic, see the examples below, replace `torchcraft_dir` argument with your torchcraft directory location

- IC3Net

```
python -u main.py --env_name starcraft --task_type explore --nagents 10 --num_epochs 1000 --hid_size 128 --lrate 0.002 --max_steps 60 --nprocesses 16 --torchcraft_dir=~/Public/TorchCraft --frame_skip 8 --nenemies 1 --our_unit_type 34 --enemy_unit_type 34 --init_range_end 150 --ic3net --recurrent --rnn_type LSTM --detach_gap 10 --stay_near_enemy --explore_vision 10 --step_size 16
```

- CommNet

```
python -u main.py --env_name starcraft --task_type explore --nagents 10 --num_epochs 1000 --hid_size 128 --lrate 0.002 --max_steps 60 --nprocesses 16 --torchcraft_dir=~/Public/TorchCraft --frame_skip 8 --nenemies 1 --our_unit_type 34 --enemy_unit_type 34 --init_range_end 150 --commnet --recurrent --rnn_type LSTM --detach_gap 10 --stay_near_enemy --explore_vision 10 --step_size 16
```

- IRIC
```
python -u main.py --env_name starcraft --task_type explore --nagents 10 --num_epochs 1000 --hid_size 128 --lrate 0.002 --max_steps 60 --nprocesses 16 --torchcraft_dir=~/Public/TorchCraft --frame_skip 8 --nenemies 1 --our_unit_type 34 --enemy_unit_type 34 --init_range_end 150 --mean_ratio 0 --recurrent --rnn_type LSTM --detach_gap 10 --stay_near_enemy --explore_vision 10 --step_size 16
```

- IC
```
python -u main.py --env_name starcraft --task_type explore --nagents 10 --num_epochs 1000 --hid_size 128 --lrate 0.002 --max_steps 60 --nprocesses 16 --torchcraft_dir=~/Public/TorchCraft --frame_skip 8 --nenemies 1 --our_unit_type 34 --enemy_unit_type 34 --init_range_end 150 --recurrent --rnn_type LSTM --detach_gap 10 --stay_near_enemy --explore_vision 10 --step_size 16
```

For 75x75, set `--init_range_end` to 175.

For Combat version:

- IC3Net
```
python -u main.py --env_name starcraft --task_type combat --nagents 10 --num_epochs 1000 --hid_size 128 --lrate 0.002 --max_steps 60 --nprocesses 16 --torchcraft_dir=~/Public/TorchCraft --frame_skip 8 --nenemies 3 --our_unit_type 0 --enemy_unit_type 65 --init_range_end 150 --ic3net --recurrent --rnn_type LSTM --detach_gap 10 --explore_vision 10 --step_size 16
```

- CommNet

```
python -u main.py --env_name starcraft --task_type combat --nagents 10 --num_epochs 1000 --hid_size 128 --lrate 0.002 --max_steps 60 --nprocesses 16 --torchcraft_dir=~/Public/TorchCraft --frame_skip 8 --nenemies 3 --our_unit_type 0 --enemy_unit_type 65 --init_range_end 150 --commnet --recurrent --rnn_type LSTM --detach_gap 10 --explore_vision 10 --step_size 16
```


- IRIC
```
python -u main.py --env_name starcraft --task_type combat --nagents 10 --num_epochs 1000 --hid_size 128 --lrate 0.002 --max_steps 60 --nprocesses 16 --torchcraft_dir=~/Public/TorchCraft --frame_skip 8 --nenemies 3 --our_unit_type 0 --enemy_unit_type 65 --init_range_end 150 --mean_ratio 0 --recurrent --rnn_type LSTM --detach_gap 10 --explore_vision 10 --step_size 16
```

- IC
```
python -u main.py --env_name starcraft --task_type combat --nagents 10 --num_epochs 1000 --hid_size 128 --lrate 0.002 --max_steps 60 --nprocesses 16 --torchcraft_dir=~/Public/TorchCraft --frame_skip 8 --nenemies 3 --our_unit_type 0 --enemy_unit_type 65 --init_range_end 150 --recurrent --rnn_type LSTM --detach_gap 10 --explore_vision 10 --step_size 16
```

## Contributors

- Amanpreet Singh ([@apsdehal](https://github.com/apsdehal))
- Tushar Jain ([@tshrjn](https://github.com/tshrjn))
- Sainbayar Sukhbaatar ([@tesatory](https://github.com/tesatory))

## License

Code is available under MIT license.
