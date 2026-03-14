import torch
import os
import torch.functional as F
from network.ppo_net import PPOActor
from network.ppo_net import PPOCritic
from torch.distributions import Categorical

# --- 引入自定义探索模块 (ARG) ---
from network.b_net import GlobalRewardPredictor
from network.arg_net import Time_Agent_Transformer
import wandb
import copy

class MAPPO:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        actor_input_shape = self.obs_shape
        critic_input_shape = self._get_critic_input_shape()

        if args.last_action:
            actor_input_shape += self.n_actions
        if args.reuse_network:
            actor_input_shape += self.n_agents
        self.args = args

        self.policy_rnn = PPOActor(actor_input_shape, args)
        self.eval_critic = PPOCritic(critic_input_shape, self.args)

        # =========================================================
        # 1. 初始化 ARG 相关网络
        # =========================================================
        self.b_net = GlobalRewardPredictor(
            state_dim=self.state_shape,
            n_agents=self.n_agents,
            n_actions=self.n_actions,
            hidden_dim=128
        )

        self.b_net_target = copy.deepcopy(self.b_net)
        # 冻结 Target 网络的梯度计算
        for param in self.b_net_target.parameters():
            param.requires_grad = False
        self.b_net_target.eval()


        self.arg_net = Time_Agent_Transformer(
            emb=self.obs_shape,
            heads=getattr(args, "n_heads", 4),
            depth=getattr(args, "arel_depth", 2),
            seq_length=args.episode_limit,
            n_agents=self.n_agents,
            agent=True,
            comp=True
        )

        if self.args.use_gpu:
            self.policy_rnn.cuda()
            self.eval_critic.cuda()
            self.b_net.cuda()
            self.b_net_target.cuda()
            self.arg_net.cuda()

        self.model_dir = args.model_dir + '/' + args.alg + '/' + args.map

        self.ac_parameters = list(self.policy_rnn.parameters()) + list(self.eval_critic.parameters())

        if args.optimizer == "RMS":
            self.ac_optimizer = torch.optim.RMSprop(self.ac_parameters, lr=args.lr)
            self.b_optimizer = torch.optim.RMSprop(self.b_net.parameters(), lr=args.lr)
            self.arg_optimizer = torch.optim.RMSprop(self.arg_net.parameters(), lr=getattr(args, "lr_arel", 1e-4))
        elif args.optimizer == "Adam":
            self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=args.lr)
            self.b_optimizer = torch.optim.Adam(self.b_net.parameters(), lr=args.lr)
            self.arg_optimizer = torch.optim.Adam(self.arg_net.parameters(), lr=getattr(args, "lr_arel", 1e-4))

        self.args = args
        self.policy_hidden = None
        self.eval_critic_hidden = None
        
        # [修复 1] 引入底层独立的计数器，彻底解耦 train_step 带来的数学隐患
        self.learn_calls = 0 

    def _get_critic_input_shape(self):
        input_shape = self.state_shape + self.obs_shape + self.n_agents
        return input_shape

    def learn(self, batch, max_episode_len, train_step, time_steps=0):
        # 每次调用 learn，计数器加 1
        self.learn_calls += 1 
        
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
                
        u, r, avail_u, terminated, s = batch['u'], batch['r'],  batch['avail_u'], batch['terminated'], batch['s']
        obs = batch['o']

        mask = (1 - batch["padded"].float())

        if self.args.use_gpu:
            u, mask, r, terminated, s = u.cuda(), mask.cuda(), r.cuda(), terminated.cuda(), s.cuda()
            obs = obs.cuda()

        # =====================================================================
        # 阶段 A: 提取并构造联合动作 One-hot
        # =====================================================================
        B, T, N, _ = u.shape
        u_onehot = torch.zeros(B, T, N, self.n_actions, device=u.device)
        u_onehot.scatter_(3, u.long(), 1.0)

        # =====================================================================
        # 阶段 B: ARG 模块计算与优化 (独立于 PPO 循环执行一次)
        # =====================================================================
        # B.1 Network B 拟合与局部惊奇度计算
        pred_r = self.b_net(s, u_onehot) # [B, T, 1]
        loss_b = (((pred_r - r) ** 2) * mask).sum() / mask.sum()
        
        self.b_optimizer.zero_grad()
        loss_b.backward()
        torch.nn.utils.clip_grad_norm_(self.b_net.parameters(), self.args.grad_norm_clip)
        self.b_optimizer.step()

        with torch.no_grad():
            pred_r_target = self.b_net_target(s, u_onehot)
            
        local_surprise = ((r - pred_r_target.detach()) ** 2) * mask # [B, T, 1]
        E_total = local_surprise.sum(dim=1) # [B, 1] 回合总惊奇度
        T_mask = mask.sum(dim=1) # [B, 1]
        
        # B.2 ARG 重分配惊奇度
        obs_arg = obs.transpose(1, 2).contiguous() # [B, N, T, Obs_Dim]
        _, x_time_wise = self.arg_net(obs_arg) 
        e_predict = x_time_wise.unsqueeze(-1) * mask # [B, T, 1] 稠密的内在奖励
        
        sum_e_predict = e_predict.sum(dim=1)
        loss_r_arg = (((sum_e_predict - E_total) ** 2) / T_mask).mean()
        
        e_mean = sum_e_predict / T_mask
        variance_per_step = ((e_predict - e_mean.unsqueeze(1)) ** 2) * mask
        loss_v_arg = (variance_per_step.sum(dim=1) / T_mask).mean()
        
        omega = getattr(self.args, "arel_omega", 20.0) 
        loss_arg = loss_r_arg + omega * loss_v_arg
        
        self.arg_optimizer.zero_grad()
        loss_arg.backward()
        torch.nn.utils.clip_grad_norm_(self.arg_net.parameters(), self.args.grad_norm_clip)
        self.arg_optimizer.step()
        
        # B.3 融合外部奖励与内在好奇心
        beta = getattr(self.args, "beta_curiosity", 0.1)
        r_mixed = r + beta * e_predict.detach()
        r = r_mixed

        if getattr(self.args, "use_wandb", False) and wandb.run is not None:
            wandb.log({
                "ARG/loss_b_net": loss_b.item(),
                "ARG/arg_loss_reg": loss_r_arg.item(),
                "ARG/arg_loss_var": loss_v_arg.item(),
                "ARG/mean_E_total_per_step": (E_total.sum() / T_mask.sum()).item(),
                "ARG/mean_intrinsic_reward": (e_predict.sum() / T_mask.sum()).item()
            }, step=time_steps)
        # =====================================================================

        mask = mask.repeat(1, 1, self.n_agents)
        r = r.repeat(1, 1, self.n_agents)
        terminated = terminated.repeat(1, 1, self.n_agents)

        old_values, _ = self._get_values(batch, max_episode_len)
        old_values = old_values.squeeze(dim=-1)
        old_action_prob = self._get_action_prob(batch, max_episode_len)

        old_dist = Categorical(old_action_prob)
        old_log_pi_taken = old_dist.log_prob(u.squeeze(dim=-1))
        old_log_pi_taken[mask == 0] = 0.0

        for _ in range(self.args.ppo_n_epochs):
            self.init_hidden(episode_num)

            values, target_values = self._get_values(batch, max_episode_len)
            values = values.squeeze(dim=-1)

            returns = torch.zeros_like(r)
            deltas = torch.zeros_like(r)
            advantages = torch.zeros_like(r)

            prev_return = 0.0
            prev_value = 0.0
            prev_advantage = 0.0
            for transition_idx in reversed(range(max_episode_len)):
                returns[:,transition_idx] = r[:,transition_idx] + self.args.gamma * prev_return * (1-terminated[:,transition_idx]) * mask[:, transition_idx]
                deltas[:,transition_idx] = r[:,transition_idx] + self.args.gamma * prev_value * (1-terminated[:,transition_idx]) * mask[:, transition_idx]\
                                           - values[:, transition_idx]
                advantages[:,transition_idx] = deltas[:,transition_idx] + self.args.gamma * self.args.lamda * prev_advantage * (1-terminated[:,transition_idx]) * mask[:, transition_idx]

                prev_return = returns[:,transition_idx]
                prev_value = values[:,transition_idx]
                prev_advantage = advantages[:,transition_idx]

            advantages = (advantages - advantages.mean()) / ( advantages.std() + 1e-8)
            advantages = advantages.detach()

            action_prob = self._get_action_prob(batch, max_episode_len)
            dist = Categorical(action_prob)
            log_pi_taken = dist.log_prob(u.squeeze(dim=-1))
            log_pi_taken[mask == 0] = 0.0

            ratios = torch.exp(log_pi_taken - old_log_pi_taken.detach())

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param) * advantages

            entropy = dist.entropy()
            entropy[mask == 0] = 0.0

            policy_loss = torch.min(surr1, surr2) + self.args.entropy_coeff * entropy
            policy_loss = - (policy_loss * mask).sum() / mask.sum()

            error_clip = torch.clamp(values - old_values.detach(), -self.args.clip_param, self.args.clip_param) + old_values.detach() - returns
            error_original = values - returns

            value_loss = 0.5 * torch.max(error_original**2, error_clip**2)
            value_loss = (mask * value_loss).sum() / mask.sum()

            loss = policy_loss + value_loss

            self.ac_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac_parameters, self.args.grad_norm_clip)
            self.ac_optimizer.step()
            
        # =====================================================================
        # [修复 2] 移出 PPO 循环，依据底层调用次数平稳更新 Target 网络
        # =====================================================================
        target_update_cycle = getattr(self.args, "target_update_cycle", 10) 
        if self.learn_calls % target_update_cycle == 0:
            self.b_net_target.load_state_dict(self.b_net.state_dict())
            
        if getattr(self.args, "use_wandb", False) and wandb.run is not None:
            wandb.log({
                "MAPPO/policy_loss": policy_loss.item(),
                "MAPPO/value_loss": value_loss.item(),
                "train_steps": train_step
            })

    def _get_critic_inputs(self, batch, transition_idx, max_episode_len):
        obs, obs_next, s, s_next = batch['o'][:, transition_idx], batch['o_next'][:, transition_idx],\
                                   batch['s'][:, transition_idx], batch['s_next'][:, transition_idx]
        s = s.unsqueeze(1).expand(-1, self.n_agents, -1)
        s_next = s_next.unsqueeze(1).expand(-1, self.n_agents, -1)
        episode_num = obs.shape[0]

        inputs, inputs_next = [], []
        inputs.append(s)
        inputs_next.append(s_next)
        inputs.append(obs)
        inputs_next.append(obs_next)
        inputs.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))
        inputs_next.append(torch.eye(self.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        inputs = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs_next], dim=1)

        return inputs, inputs_next

    def _get_values(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        v_evals, v_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_critic_inputs(batch, transition_idx, max_episode_len)
            if self.args.use_gpu:
                inputs = inputs.cuda()
                self.eval_critic_hidden = self.eval_critic_hidden.cuda()

            v_eval, self.eval_critic_hidden = self.eval_critic(inputs, self.eval_critic_hidden)
            v_eval = v_eval.view(episode_num, self.n_agents, -1)
            v_evals.append(v_eval)

        v_evals = torch.stack(v_evals, dim=1)
        return v_evals, v_targets

    def _get_actor_inputs(self, batch, transition_idx):
        obs, u_onehot = batch['o'][:, transition_idx], batch['u_onehot'][:]
        episode_num = obs.shape[0]
        inputs = []
        inputs.append(obs)

        if self.args.last_action:
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1))

        inputs = torch.cat([x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1)

        return inputs

    def _get_action_prob(self, batch, max_episode_len):
        episode_num = batch['o'].shape[0]
        avail_actions = batch['avail_u']
        action_prob = []
        for transition_idx in range(max_episode_len):
            inputs = self._get_actor_inputs(batch, transition_idx)
            if self.args.use_gpu:
                inputs = inputs.cuda()
                self.policy_hidden = self.policy_hidden.cuda()
            outputs, self.policy_hidden = self.policy_rnn(inputs, self.policy_hidden)
            outputs = outputs.view(episode_num, self.n_agents, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)
            action_prob.append(prob)

        action_prob = torch.stack(action_prob, dim=1).cpu()
        action_prob = action_prob + 1e-10

        action_prob[avail_actions == 0] = 0.0
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True)
        action_prob[avail_actions == 0] = 0.0
        
        action_prob = action_prob + 1e-10

        if self.args.use_gpu:
            action_prob = action_prob.cuda()
        return action_prob

    def init_hidden(self, episode_num):
        self.policy_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))
        self.eval_critic_hidden = torch.zeros((episode_num, self.n_agents, self.args.rnn_hidden_dim))

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_critic.state_dict(), self.model_dir + '/' + num + '_critic_params.pkl')
        torch.save(self.policy_rnn.state_dict(),  self.model_dir + '/' + num + '_rnn_params.pkl')
        
        torch.save(self.b_net.state_dict(),  self.model_dir + '/' + num + '_b_net_params.pkl')
        torch.save(self.arg_net.state_dict(),  self.model_dir + '/' + num + '_arg_net_params.pkl')