import argparse
import torch

"""
Here are the param for the training
"""

def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--difficulty', type=str, default='6', help='the difficulty of the game')
    parser.add_argument('--game_version', type=str, default='latest', help='the version of the game')
    parser.add_argument('--map', type=str, default='3m', help='the map of the game')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--step_mul', type=int, default=8, help='how many steps to make an action')
    parser.add_argument('--replay_dir', type=str, default='', help='the directory of save the replay')
    parser.add_argument('--alg', type=str, default='mappo', help='the algorithm to train the agent')
    parser.add_argument('--last_action', type=bool, default=True, help='whether to use the last action to choose action')
    parser.add_argument('--reuse_network', type=bool, default=True, help='whether to use one network for all agents')
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor')
    parser.add_argument('--optimizer', type=str, default="Adam", help='the optimizer')
    parser.add_argument('--model_dir', type=str, default='./model', help='the model directory of the policy')
    parser.add_argument('--result_dir', type=str, default='./model', help='the result directory of the policy')
    parser.add_argument('--learn', type=bool, default=True, help='whether to train the model')
    parser.add_argument('--threshold', type=int, default=19, help='the threshold to judge whether win')
    parser.add_argument('--evaluate_cycle', type=int, default=10000, help='how often to evaluate the model')
    parser.add_argument('--n_steps', type=int, default=2050000, help='total time steps')
    parser.add_argument('--evaluate_epoch', type=int, default=32, help='number of the epoch to evaluate the agent')

    # WandB setting
    parser.add_argument('--use_wandb', type=bool, default=True, help='whether to use wandb')
    parser.add_argument('--wandb_project', type=str, default='SMAC_MAPPO_ARG', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity name (username or team)')

    # =====================================================================
    # [新增] ARG & Curiosity 探索模块专属超参数
    # =====================================================================
    parser.add_argument('--lr_b', type=float, default=1e-4, help='Network B (真实奖励预测器) 的学习率')
    parser.add_argument('--lr_arel', type=float, default=1e-4, help='ARG (注意力重分配网络) 的学习率')
    parser.add_argument('--arel_omega', type=float, default=5.0, help='方差正则化权重，默认 5.0 以避免奖励过度平滑')
    parser.add_argument('--beta_curiosity', type=float, default=0.1, help='0.1内在探索奖励的融合系数 (beta)')
    parser.add_argument('--arel_depth', type=int, default=2, help='ARG Transformer 块的层数')
    parser.add_argument('--n_heads', type=int, default=4, help='ARG Transformer 的多头注意力头数')
    parser.add_argument('--target_update_cycle', type=int, default=20, help='b_target网络更新间隔')

    args = parser.parse_args()
    return args

def get_mixer_args(args):
    args.use_gpu = torch.cuda.is_available()

    args.rnn_hidden_dim = 64
    args.lr = 5e-4
    args.lr_actor = 5e-4
    args.lr_critic = 5e-4
    args.train_steps = 1

    # how often to save the model
    args.save_cycle = 1000000

    # how often to update the target_net
    # args.target_update_cycle = 200
    args.grad_norm_clip = 10

    args.n_episodes = 32
    args.ppo_n_epochs = 15
    args.lamda = 0.95
    args.clip_param = 0.2
    args.entropy_coeff = 0.01

    return args