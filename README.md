tensorboard --logdir runs/ --port 6006

python ppo.py --start-level 3 --max-level 3 --total-timesteps 200000 --exp-name level3-isolated
