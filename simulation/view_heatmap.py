import numpy as np
import matplotlib.pyplot as plt

def view_heatmap():
    print("正在加载 3D 数据库，请稍候...")
    # 1. 加载你刚刚生成的 5D 矩阵
    # 维度: [X索引(11), Y索引(11), Z索引(7), Rx(125), Ry(125)]
    database = np.load('paris_pg_database_3d.npy')
    print(f"数据库加载成功！形状为: {database.shape}")

    # 2. 挑选一个你感兴趣的基站位置来“偷窥”
    # 比如我们选地图最中心的一个点 (X的第6个，Y的第6个)
    x_idx = 5  
    y_idx = 5  
    
    # 我们看看最高的高度 (索引6，对应35米) 和 最低的高度 (索引0，对应5米) 的区别
    z_idx_high = 6  
    z_idx_low = 0   

    # 3. 提取出这两张 2D 热力图
    heatmap_high = database[x_idx, y_idx, z_idx_high, :, :]
    heatmap_low = database[x_idx, y_idx, z_idx_low, :, :]

    # ✅ 物理显示修正：
    # 因为我们库里存的是极小的“线性增益”，如果直接画图，画面可能几乎全黑。
    # 为了人眼能看清楚，我们在画图时临时把它转换回 dB 单位。
    # 加 1e-15 是为了防止给 0 取对数报错。
    heatmap_high_db = 10 * np.log10(heatmap_high + 1e-15)
    heatmap_low_db = 10 * np.log10(heatmap_low + 1e-15)

    # 4. 画出对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 画低处 (5米) 的热力图
    im1 = axes[0].imshow(heatmap_low_db, cmap='viridis', origin='lower')
    axes[0].set_title(f"AP Height: 5m (Index: {z_idx_low})")
    fig.colorbar(im1, ax=axes[0], label='Path Gain (dB)')

    # 画高处 (35米) 的热力图
    im2 = axes[1].imshow(heatmap_high_db, cmap='viridis', origin='lower')
    axes[1].set_title(f"AP Height: 35m (Index: {z_idx_high})")
    fig.colorbar(im2, ax=axes[1], label='Path Gain (dB)')

    plt.suptitle("3D Cell-Free Database Verification", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    view_heatmap()