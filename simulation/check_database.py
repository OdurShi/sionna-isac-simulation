import numpy as np
import matplotlib.pyplot as plt

def fast_diagnostic(db_path='paris_pg_database_3d.npy', pos_path='candidate_positions_3d.npy'):
    print(f"🔍 正在加载数据库: {db_path} ...")
    db = np.load(db_path)
    positions = np.load(pos_path)
    
    print(f"📊 数据库形状: {db.shape}")
    print(f"📍 候选位置数量: {positions.shape[0]}")
    
    # 将 5D 矩阵展平为前三个维度合并，方便逐个基站位置进行检查
    # shape 变成 (847, 125, 125)
    num_candidates = db.shape[0] * db.shape[1] * db.shape[2]
    db_flat = db.reshape(num_candidates, 125, 125)
    
    # ---------------------------------------------------------
    # 诊断 1：检查“死区” (全零或极弱信号的基站位置)
    # ---------------------------------------------------------
    print("\n--- 诊断 1: 盲区与死点检测 ---")
    # 计算每个基站覆盖范围内的总能量 (如果信号太弱，说明基站可能建在建筑物内部了)
    total_energy_per_tx = np.sum(db_flat, axis=(1, 2))
    
    # 找出能量为 0 或者极低的异常点
    dead_zones = np.where(total_energy_per_tx < 1e-15)[0] 
    print(f"⚠️ 发现 {len(dead_zones)} 个基站位置的总覆盖能量极低 (可能是非法位置/卡在墙里)")
    
    if len(dead_zones) > 0:
        print("   前 5 个问题坐标:")
        for idx in dead_zones[:5]:
            print(f"   - 索引 {idx}: 坐标 {positions[idx]}")

    # ---------------------------------------------------------
    # 诊断 2：检查“散斑”和射线数量不足 (稀疏度检测)
    # ---------------------------------------------------------
    print("\n--- 诊断 2: 射线覆盖稀疏度检测 ---")
    # 计算每个热力图 (125x125) 中，完全没有收到信号的像素点比例
    zero_pixel_ratio = np.sum(db_flat == 0, axis=(1, 2)) / (125 * 125)
    
    # 如果一张热力图超过 80% 的区域是 0，说明要么被严重遮挡，要么就是射线数量太少没打满
    sparse_maps = np.where(zero_pixel_ratio > 0.80)[0]
    print(f"⚠️ 发现 {len(sparse_maps)} 个热力图的无效覆盖面积超过 80% (可能需要增加 samples_per_tx)")
    
    # ---------------------------------------------------------
    # 诊断 3：全局数值统计与极端值检测
    # ---------------------------------------------------------
    print("\n--- 诊断 3: 全局数值健康度 ---")
    print(f"   📈 最大信道增益 (Linear): {np.max(db)}")
    print(f"   📉 最小信道增益 (Linear): {np.min(db)}")
    print(f"   🧮 平均信道增益 (Linear): {np.mean(db)}")
    
    # 检查是否有 NaN 或无穷大 (这会让后续的 SAC 神经网络直接崩溃)
    if np.isnan(db).any() or np.isinf(db).any():
        print("❌ 严重警告：数据库中包含 NaN 或 Inf 异常值！必须清理！")
    else:
        print("✅ 数值纯净度检查通过 (无 NaN/Inf)。")

    # ---------------------------------------------------------
    # 诊断 4：随机抽查可视化 (非常关键！)
    # ---------------------------------------------------------
    print("\n--- 诊断 4: 随机抽取 3 个正常样本进行可视化 ---")
    # 避开刚才找到的 dead_zones，随机挑几个正常的看看长什么样
    valid_indices = np.setdiff1d(np.arange(num_candidates), dead_zones)
    
    if len(valid_indices) >= 3:
        sample_idx = np.random.choice(valid_indices, 3, replace=False)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, idx in enumerate(sample_idx):
            # 转换为对数域 (dB) 方便观察大尺度衰落，加 1e-15 防止 log10(0)
            heatmap_db = 10 * np.log10(db_flat[idx] + 1e-15) 
            
            im = axes[i].imshow(heatmap_db, cmap='jet', vmin=-150, vmax=-50)
            axes[i].set_title(f"TX {idx} (Z={positions[idx][2]}m)\nAvg: {np.mean(heatmap_db):.1f} dB")
            fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ 没有足够的有效热力图可供可视化。")

if __name__ == "__main__":
    fast_diagnostic()