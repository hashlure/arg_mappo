import torch
import torch.nn as nn
import torch.nn.functional as F

class Time_Agent_Transformer(nn.Module):
    """
    ARG 核心网络：严格按照 AREL 论文实现的双重注意力架构
    包含：MLP 特征压缩 -> 时间因果注意力 -> 智能体交互注意力 -> Deep Sets 信用分配
    """
    def __init__(self, emb, heads, depth, seq_length, n_agents, agent=True, comp=True):
        super().__init__()
        self.n_agents = n_agents
        self.seq_length = seq_length
        self.hidden_dim = 64
        self.use_agent_attn = agent
        
        # ==========================================
        # 1. 增强版特征压缩 (MLP Encoder)
        # ==========================================
        # 将高维冷冰冰的原始物理特征，映射到富含语义且统一量纲的隐空间
        self.obs_proj = nn.Sequential(
            nn.Linear(emb, 128),
            nn.LayerNorm(128), # 统一量纲，防止某些物理量绝对值过大冲刷梯度
            nn.ReLU(),
            nn.Linear(128, self.hidden_dim)
        )
        
        # 位置编码 (Positional Embedding)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, seq_length, self.hidden_dim))
        
        # ==========================================
        # 2. 时间注意力层 (Temporal Attention)
        # ==========================================
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, 
            nhead=heads, 
            dim_feedforward=self.hidden_dim * 2, 
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=depth)
        
        # ==========================================
        # 3. 智能体注意力层 (Agent Attention)
        # ==========================================
        if self.use_agent_attn:
            agent_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim, 
                nhead=heads, 
                dim_feedforward=self.hidden_dim * 2, 
                batch_first=True
            )
            # 智能体交互通常一层就足够提取当前帧的拓扑关系
            self.agent_transformer = nn.TransformerEncoder(agent_layer, num_layers=1) 
        
        # ==========================================
        # 4. 信用分配多层感知机 (Deep Sets)
        # ==========================================
        self.g1 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim), 
            nn.LayerNorm(self.hidden_dim), # 关键保护：稳定方差约束的梯度
            nn.ReLU()
        )
        self.g2 = nn.Sequential(
            nn.Linear(self.hidden_dim, 32), 
            nn.ReLU(), 
            nn.Linear(32, 1)
        )

    def forward(self, obs_arg):
        # 初始输入: [Batch, N_agents, Time, Obs_Dim]
        B, N, T, _ = obs_arg.shape
        
        # ------------------------------------------
        # 阶段 A: 特征提炼与定位
        # ------------------------------------------
        # 1. 投影特征: [B, N, T, H]
        x = self.obs_proj(obs_arg) 
        
        # 2. 注入时间位置编码: [B, N, T, H]
        x = x + self.pos_embedding[:, :, :T, :] 
        
        # ------------------------------------------
        # 阶段 B: 时间因果注意力 (审视历史)
        # ------------------------------------------
        # 视角切换：把 B 和 N 压扁，平行看 (B*N) 条个人时间线
        x_tem = x.view(B * N, T, self.hidden_dim) # [B*N, T, H]
        
        # 生成因果掩码 (下三角矩阵，防止穿越看未来)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device) * float('-inf'), diagonal=1)
        
        # 通过时间 Transformer: [B*N, T, H]
        tem_out = self.temporal_transformer(x_tem, mask=causal_mask) 
        
        # 恢复出四维形状: [B, N, T, H]
        tem_out = tem_out.view(B, N, T, self.hidden_dim)
        
        # ------------------------------------------
        # 阶段 C: 智能体注意力 (寻找化学反应)
        # ------------------------------------------
        if self.use_agent_attn:
            # 【核心数学变换】：必须先 transpose 再 view！
            # 将 [B, N, T, H] 转置为 [B, T, N, H]
            # 物理意义：把整个游戏按“时间帧”切片，每一帧里包含 N 个人的特征
            x_agt = tem_out.transpose(1, 2).contiguous() 
            
            # 压扁 B 和 T，平行看 (B*T) 张快照，每张快照里 N 个人互相对视
            x_agt = x_agt.view(B * T, N, self.hidden_dim) # [B*T, N, H]
            
            # 通过智能体 Transformer (同一帧内大家都在场，不需要 Mask)
            agt_out = self.agent_transformer(x_agt) # [B*T, N, H]
            
            # 恢复为时空终极特征矩阵 Z: [B, T, N, H]
            Z = agt_out.view(B, T, N, self.hidden_dim)
        else:
            # 如果不启用智能体注意力，也必须把维度换成以 Time 为主轴
            Z = tem_out.transpose(1, 2).contiguous() # [B, T, N, H]
            
        # ------------------------------------------
        # 阶段 D: Deep Sets 坍缩与奖励生成
        # ------------------------------------------
        # 对终极特征 Z 独立打分: [B, T, N, H]
        g1_out = self.g1(Z) 
        
        # 置换不变性核心：抹平智能体维度 (N 维在第 2 位，即 dim=2)
        sum_agents = g1_out.sum(dim=2) # [B, T, H]
        
        # 最终预测的该时间步总体奖励/惊奇度
        r_pred_time_wise = self.g2(sum_agents).squeeze(-1) # [B, T]
        
        return None, r_pred_time_wise