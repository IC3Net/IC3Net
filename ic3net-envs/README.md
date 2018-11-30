# IC3Net Environments

This repository contains gym environments for tasks used in paper for IC3Net except starcraft. Namely, this repository contains:

- Traffic Junction Environment
- Predator Prey Environments
- Sanity check number pairs and levers environment will be added later.

## Running

Run `python setup.py develop` in the locally cloned repository.
Next, run `python example/random_agent.py` for a random agent playing with Traffic Junction environment.

Note that, you can use `--display` flag to see the actual environment being rendered on console. You might not see anything as it is action and execution are very fast in case of a random agent.

## License

Code for this project is available under MIT license.
