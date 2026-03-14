import torch as th
import torch.nn as nn
import torch.nn.functional as F

class GlobalRewardPredictor(nn.Module):
    """
    Network B: 利用全局状态和联合动作预测环境原始单步密集奖励。
    """
    def __init__(self, state_dim, n_agents, n_actions, hidden_dim=128):
        super(GlobalRewardPredictor, self).__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        
        # 输入维度: 全局状态维度 + 扁平化后的联合动作维度
        input_dim = state_dim + (n_agents * n_actions)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, state, actions_onehot):
        # state: [B, T, state_dim]
        # actions_onehot: [B, T, n_agents, n_actions]
        B, T, _, _ = actions_onehot.shape
        
        # 展平联合动作 [B, T, n_agents * n_actions]
        actions_flat = actions_onehot.view(B, T, -1) 
        
        # 拼接全局状态与联合动作 [B, T, state_dim + n_agents * n_actions]
        x = th.cat([state, actions_flat], dim=-1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        pred_r = self.out(x) # [B, T, 1]
        
        return pred_r