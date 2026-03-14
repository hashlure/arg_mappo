import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Time_Agent_Transformer(nn.Module):
    """
    ARG 核心网络：包含智能体注意力与时间因果注意力
    """
    def __init__(self, emb, heads, depth, seq_length, n_agents, agent=True, comp=True):
        super().__init__()
        self.n_agents = n_agents
        self.seq_length = seq_length
        self.hidden_dim = 64
        
        # 特征降维/升维
        self.obs_proj = nn.Linear(emb, self.hidden_dim)
        
        # 位置编码 (Positional Embedding)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, seq_length, self.hidden_dim))
        
        # Transformer 层 (简化的多层实现)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, 
            nhead=heads, 
            dim_feedforward=self.hidden_dim * 2, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 信用分配多层感知机 (保证置换不变性的 Deep Sets 架构)
        self.g1 = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU())
        self.g2 = nn.Sequential(nn.Linear(self.hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, obs_arg):
        # obs_arg: [B, N, T, Obs_Dim]
        B, N, T, _ = obs_arg.shape
        
        # 1. 投影特征
        x = self.obs_proj(obs_arg) # [B, N, T, H]
        
        # 2. 注入时间位置编码 (广播到所有 Agent)
        x = x + self.pos_embedding[:, :, :T, :] 
        
        # 3. 时间与智能体维度的融合注意力 (为了简化并保持稳定，将 B和N 展平处理时序，再融合)
        x_flat = x.view(B * N, T, self.hidden_dim) # [B*N, T, H]
        
        # 生成因果掩码 (下三角矩阵，防止看到未来)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        
        # 通过 Transformer
        attn_out = self.transformer(x_flat, mask=causal_mask) # [B*N, T, H]
        attn_out = attn_out.view(B, N, T, self.hidden_dim)
        
        # 4. 智能体维度的置换不变性计算 (Deep Sets 操作: g2(sum(g1(x))))
        # 对每一个时间步的 N 个智能体求和
        g1_out = self.g1(attn_out) # [B, N, T, H]
        sum_agents = g1_out.sum(dim=1) # [B, T, H] 抹平智能体维度
        
        # 最终预测的该时间步总体奖励/惊奇度
        r_pred_time_wise = self.g2(sum_agents).squeeze(-1) # [B, T]
        
        return None, r_pred_time_wise