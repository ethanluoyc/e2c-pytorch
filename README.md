# Embed to Control implementation in PyTorch

Paper can be found here: <https://arxiv.org/abs/1506.07365>

You will need a patched version of OpenAI Gym in order to generate the
dataset. See <https://github.com/ethanluoyc/gym/tree/pendulum_internal>

For the planar task, we use code from. The source code of the repository
has been modified for our needs and included under `e2c/e2c_tf`.

## What's included ?
* E2C model, VAE and AE baselines. Allow configuration for different
network architecture for the different setups (see Appendix of the paper).

## TODO
* Documentation, tests... (Soon to follow)
