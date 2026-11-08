#!/bin/bash

# Cora
python training.py --dataset cora --device 0 --hidden 64 --num_partitions 15 --lr 0.1 --prop_g_lr 0.5 --prop_f_lr 0.1 --weight_decay 0.0005 --prop_g_wd 0.0005 --prop_f_wd 0.05 --dropout 0.5 --dprate 0.0 --K 10

# Citeseer
python training.py --dataset citeseer --device 1 --hidden 128 --num_partitions 5 --lr 0.05 --prop_g_lr 0.5 --prop_f_lr 0.05 --weight_decay 0.05 --prop_g_wd 0.0005 --prop_f_wd 0.05 --dropout 0.0 --dprate 0.5 --K 10

# Pubmed
python training.py --dataset pubmed --device 2 --hidden 128 --num_partitions 10 --lr 0.01 --prop_g_lr 0.5 --prop_f_lr 0.5 --weight_decay 0.0 --prop_g_wd 0.0 --prop_f_wd 0.05 --dropout 0.5 --dprate 0.0 --K 10

# Roman Empire (50%/25%/25% split)
python training.py --dataset roman_empire --device 3 --hidden 128 --num_partitions 5 --lr 0.01 --prop_g_lr 0.05 --prop_f_lr 0.05 --weight_decay 0.0 --prop_g_wd 0.0 --prop_f_wd 0.0005 --dropout 0.5 --dprate 0.0 --K 10

# Amazon Ratings (50%/25%/25% split)
python training.py --dataset amazon_ratings --device 0 --hidden 128 --num_partitions 10 --lr 0.005 --prop_g_lr 0.005 --prop_f_lr 0.5 --weight_decay 0.0 --prop_g_wd 0.0005 --prop_f_wd 0.05 --dropout 0.0 --dprate 0.5

# Questions (50%/25%/25% split)
python training.py --dataset questions --device 1 --hidden 128 --num_partitions 5 --lr 0.05 --prop_g_lr 0.5 --prop_f_lr 0.5 --weight_decay 0.0005 --prop_g_wd 0.0 --prop_f_wd 0.0005 --dropout 0.0 --dprate 0.0 --K 10

# Twitch Gamer (50%/25%/25% split)
python training.py --dataset twitch-gamer --device 2 --hidden 128 --num_partitions 15 --lr 0.05 --prop_g_lr 0.05 --prop_f_lr 0.1 --weight_decay 0.0 --prop_g_wd 0.0 --prop_f_wd 0.05 --dropout 0.0 --dprate 0.5 --K 10

# Genius (50%/25%/25% split)
python training.py --dataset genius --device 3 --hidden 32 --num_partitions 15 --lr 0.1 --prop_g_lr 0.001 --prop_f_lr 0.05 --weight_decay 0.0 --prop_g_wd 0.0 --prop_f_wd 0.0005 --dropout 0.5 --dprate 0.0 --K 10

# Actor
python training.py --dataset actor --device 0 --hidden 128 --num_partitions 20 --lr 0.05 --prop_g_lr 0.005 --prop_f_lr 0.1 --weight_decay 0.0005 --prop_g_wd 0.05 --prop_f_wd 0.0005 --dropout 0.5 --dprate 0.5 --K 10

# Chameleon
python training.py --dataset chameleon --device 1 --hidden 64 --num_partitions 5 --lr 0.005 --prop_g_lr 0.05 --prop_f_lr 0.001 --weight_decay 0.0005 --prop_g_wd 0.0 --prop_f_wd 0.0005 --dropout 0.5 --dprate 0.0 --K 10

# Squirrel
python training.py --dataset squirrel --device 0 --hidden 128 --num_partitions 5 --lr 0.005 --prop_g_lr 0.05 --prop_f_lr 0.01 --weight_decay 0.0 --prop_g_wd 0.0 --prop_f_wd 0.0005 --dropout 0.5 --dprate 0.5 --K 10

# Computers
python training.py --dataset computers --device 1 --hidden 64 --num_partitions 15 --lr 0.01 --prop_g_lr 0.05 --prop_f_lr 0.01 --weight_decay 0.0005 --prop_g_wd 0.0 --prop_f_wd 0.05 --dropout 0.0 --dprate 0.5 --K 10

# Photo
python training.py --dataset photo --device 0 --hidden 64 --num_partitions 15 --lr 0.005 --prop_g_lr 0.01 --prop_f_lr 0.01 --weight_decay 0.0005 --prop_g_wd 0.0 --prop_f_wd 0.05 --dropout 0.5 --dprate 0.5 --K 10

# Penn94
python training.py --dataset penn94 --device 1 --hidden 64 --num_partitions 15 --lr 0.005 --prop_g_lr 0.01 --prop_f_lr 0.01 --weight_decay 0.0005 --prop_g_wd 0.0 --prop_f_wd 0.05 --dropout 0.5 --dprate 0.5 --K 10


