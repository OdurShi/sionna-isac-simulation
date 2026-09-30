import numpy as np
from isac_env import CellFreeISAC3DEnv

def test_environment():
    print("🛠️ 开始环境暴力压力测试...")
    env = CellFreeISAC3DEnv(M=2, N=2, K=15)
    
    state = env.reset()
    print(f"✅ Reset 成功！初始状态维度: {state.shape}")
    
    # 随便跑 5 步看看会不会崩溃
    for step in range(5):
        # 生成完全随机的连续动作 [-1, 1]
        random_action = np.random.uniform(-1, 1, env.action_space.shape[0])
        
        next_state, reward, done, info = env.step(random_action, 'max-sum')
        
        print(f"Step {step+1}:")
        print(f"  - 奖励 Reward: {reward:.4f}")
        print(f"  - 通信速率 sum_rate: {info['sum_rate']:.4f}")
        print(f"  - 感知精度 fim_det: {info['fim_det']:.4f}")
        print(f"  - 选中的 4 个基站: {info['selected_aps']}")
        
        if np.isnan(reward) or np.isinf(reward):
            print("❌ 警告：奖励值出现了 NaN 或无穷大！")
            break

if __name__ == '__main__':
    test_environment()