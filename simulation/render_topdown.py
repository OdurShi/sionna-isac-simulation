import numpy as np
import matplotlib.pyplot as plt

def main():
    print("🧠 正在加载你的物理仿真数据库...")
    database = np.load('paris_pg_database_3d.npy')
    
    print("✨ 正在聚合 121 个基站的信号，逆向提取建筑轮廓...")
    # 提取 35m 高度 (Z轴索引 6) 下的数据
    # 对前两个维度 (X, Y 坐标，共 11x11=121 个基站) 取最大值
    # 这等同于：“在这个城市的每个角落，选出能接收到的最强信号”
    max_coverage_linear = np.max(database[:, :, 6, :, :], axis=(0, 1))
    
    # 转换为 dB 单位显示
    max_coverage_db = 10 * np.log10(max_coverage_linear + 1e-15)
    
    # 开始画图
    plt.figure(figsize=(10, 10))
    # 使用 magma 配色：街道(有信号)会是明亮的黄/橙/粉色，实体建筑物内部(无信号)则是纯黑色
    plt.imshow(max_coverage_db, cmap='magma', origin='lower')
    plt.title("Paris Etoile True Layout (Reconstructed from Ray Tracing)", fontsize=16)
    plt.axis('off')
    
    save_path = "reconstructed_map.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ 完美！依靠你自己的数据逆向提取的地图已保存为：{save_path}")

if __name__ == "__main__":
    main()