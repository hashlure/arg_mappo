import numpy as np
import os
import copy
import multiprocessing as mp
from common.rollout import RolloutWorker
from agent.agent import Agents
import matplotlib.pyplot as plt
from smac.env import StarCraft2Env
import wandb
import torch
import time

# =====================================================================
# [新增] 独立运行在子进程中的采样工厂函数
# =====================================================================
def worker_process(remote, parent_remote, args):
    parent_remote.close() # 子进程不需要主进程的通信端
    
    # 【底层优化 1】强制限制每个子进程的 PyTorch CPU 线程数为 1
    # 理由：避免多个子进程同时抢占多核 CPU 导致的严重上下文切换和卡顿
    torch.set_num_threads(1)
    
    # 【底层优化】强制子进程使用 CPU 进行推理
    worker_args = copy.deepcopy(args)
    worker_args.use_gpu = False 

    # 初始化独立的星际环境与 Worker
    env = StarCraft2Env(map_name=worker_args.map,
                        step_mul=worker_args.step_mul,
                        difficulty=worker_args.difficulty,
                        game_version=worker_args.game_version,
                        replay_dir='') # 采样进程禁止保存录像，节省 I/O
    agents = Agents(worker_args)
    worker = RolloutWorker(env, agents, worker_args)

    while True:
        cmd, data = remote.recv()
        if cmd == 'sync':
            # 接收主进程传来的最新网络权重，并加载到本地
            agents.policy.policy_rnn.load_state_dict(data['rnn'])
            agents.policy.eval_critic.load_state_dict(data['critic'])
            remote.send('sync_done')
        elif cmd == 'generate':
            # 并行执行分配给该进程的 Episode 数量
            num_episodes = data
            episodes = []
            total_steps = 0
            for _ in range(num_episodes):
                ep, _, _, steps = worker.generate_episode(0)
                episodes.append(ep)
                total_steps += steps
            remote.send((episodes, total_steps))
        elif cmd == 'close':
            env.close()
            remote.close()
            break


class Runner:
    def __init__(self, env, args):
        self.env = env
        # 主进程依然保留一个环境和 Worker，专门用来做稳定的 Evaluate
        env = StarCraft2Env(map_name=args.map,
                            step_mul=args.step_mul,
                            difficulty=args.difficulty,
                            game_version=args.game_version,
                            replay_dir=args.replay_dir)
        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.args = args

        self.win_rates = []
        self.episode_rewards = []

        self.save_path = self.args.result_dir + '/' + args.alg + '/' + args.map
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # =========================================================
        # [新增] 启动多进程采样池
        # =========================================================
        # 建议在排查阶段将 n_rollout_threads 设为 4
        self.n_threads = getattr(self.args, "n_rollout_threads", 4) 
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(self.n_threads)])
        self.processes = []
        
        print(f"正在初始化 {self.n_threads} 个后台采样进程，请耐心等待...")
        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            p = mp.Process(target=worker_process, args=(work_remote, remote, args))
            p.daemon = True # 确保主程序退出时，子进程能被强制杀死
            p.start()
            
            # 【底层优化 2】错峰启动
            # 理由：SMAC 环境启动时 I/O 极高，错开 2 秒能有效防止系统卡死
            time.sleep(2.0) 
            self.processes.append(p)
            print(f"进程 {i+1}/{self.n_threads} 启动指令已发送。")

    def run(self, num=0):
        # 【底层优化 3】将 evaluate_steps 初始值设为 0（原为 -1）
        # 理由：跳过 step=0 时的初始评估，避免在环境刚建好时立刻阻塞主线程
        time_steps, train_steps, evaluate_steps = 0, 0, 0 
        
        while time_steps < self.args.n_steps:

            print('Run {}, time_steps {}'.format(num, time_steps))
            
            # 1. 评估与日志 (这部分只在主进程串行执行，保证稳定)
            if time_steps > 0 and (time_steps // self.args.evaluate_cycle >= evaluate_steps):
                win_rate, episode_reward = self.evaluate()
                self.win_rates.append(win_rate)
                self.episode_rewards.append(episode_reward)
                if getattr(self.args, "use_wandb", False):
                    wandb.log({
                        "test_win_rate": win_rate,
                        "test_episode_reward": episode_reward,
                        "time_steps": time_steps
                    })
                
                # 降低 plt 的频率，因为往硬盘写图片非常拖慢速度
                if evaluate_steps % 5 == 0:
                    self.plt(num)
                evaluate_steps += 1

            # =====================================================
            # [新增] 2. 并行采样逻辑核心
            # =====================================================
            
            # A. 提取主进程最新的网络权重，并移动到 CPU (防止进程间传递 CUDA Tensor 报错)
            state_dict_to_send = {
                'rnn': {k: v.cpu() for k, v in self.agents.policy.policy_rnn.state_dict().items()},
                'critic': {k: v.cpu() for k, v in self.agents.policy.eval_critic.state_dict().items()}
            }
            
            # B. 广播权重给所有子进程并等待确认
            for remote in self.remotes:
                remote.send(('sync', state_dict_to_send))
            for remote in self.remotes:
                remote.recv()

            # C. 均匀分配 N 个 episodes 给各个进程
            episodes_per_thread = self.args.n_episodes // self.n_threads
            remainder = self.args.n_episodes % self.n_threads
            
            for i, remote in enumerate(self.remotes):
                tasks = episodes_per_thread + (1 if i < remainder else 0)
                remote.send(('generate', tasks))

            # D. 收集所有子进程跑完的数据
            episodes = []
            for remote in self.remotes:
                eps, steps = remote.recv()
                episodes.extend(eps)
                time_steps += steps
            # =====================================================

            # 3. 高效拼接数据
            episode_batch = episodes[0]
            episodes.pop(0)
            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate((episode_batch[key], episode[key]), axis=0)

            # 4. 主进程 GPU 训练
            self.agents.train(episode_batch, train_steps, time_steps)
            train_steps += self.args.ppo_n_epochs

        # 结束前的最终评估与清理
        win_rate, episode_reward = self.evaluate()
        print('Final win_rate is ', win_rate)
        self.win_rates.append(win_rate)
        self.episode_rewards.append(episode_reward)
        self.plt(num)
        self.close_processes() # 释放资源

    def evaluate(self):
        win_number = 0
        episode_rewards = 0
        for epoch in range(self.args.evaluate_epoch):
            _, episode_reward, win_tag, _ = self.rolloutWorker.generate_episode(epoch, evaluate=True)
            episode_rewards += episode_reward
            if win_tag:
                win_number += 1
        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch

    def plt(self, num):
        plt.figure(figsize=(12, 8))
        plt.axis([0, self.args.n_steps, 0, 5000])
        plt.cla()

        plt.subplot(2, 1, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates)
        plt.xlabel('1e4 timesteps')
        plt.ylabel('win_rate')

        plt.subplot(2, 1, 2)
        plt.plot(range(len(self.episode_rewards)), self.episode_rewards)
        plt.xlabel('1e4 timesteps')
        plt.ylabel('episode_rewards')

        plt.savefig(self.save_path + '/plt.png'.format(num), format='png')
        np.save(self.save_path + '/win_rates'.format(num), self.win_rates)
        np.save(self.save_path + '/episode_rewards'.format(num), self.episode_rewards)

        plt.close()

    def close_processes(self):
        # [新增] 安全关闭所有子进程
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.processes:
            p.join()