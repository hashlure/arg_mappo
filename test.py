import os
import smac
from smac.env import StarCraft2Env

# 验证环境变量
print(f"当前 SC2PATH 路径: {os.environ.get('SC2PATH')}")

# 验证是否能找到游戏
try:
    # 尝试初始化一个最基础的地图（确保你的 Maps 文件夹里有这个地图）
    env = StarCraft2Env(map_name="3m")
    print("✅ 成功：Python 已识别 SMAC 且已连接到星际争霸II游戏核心！")
    env.close()
except Exception as e:
    print(f"❌ 失败：虽然找到了库，但游戏启动失败。错误信息: {e}")