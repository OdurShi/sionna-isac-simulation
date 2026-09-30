import os
# 屏蔽 GPU，用纯 CPU 暴力算
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import sionna
from sionna.rt import load_scene, Transmitter, PlanarArray, RadioMapSolver
import matplotlib.pyplot as plt

def main():
    print("🎬 正在加载巴黎 3D 城市模型...")
    scene = load_scene(sionna.rt.scene.etoile)

    print("📡 正在全局注册收发天线阵列...")
    scene.tx_array = PlanarArray(num_rows=8, num_cols=8, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

    print("🏗️ 正在架设 5G 基站...")
    tx = Transmitter(name="BaseStation_1", position=[0, 0, 30], orientation=[0, 0, 0])
    scene.add(tx)

    print("🚀 正在启动求解器进行光线追踪...")
    solver = RadioMapSolver()
    
    radio_map = solver(
        scene=scene, 
        cell_size=[4, 4], 
        center=[0, 0, 1.5], 
        orientation=[0, 0, 0], 
        size=[500, 500],
        samples_per_tx=int(1e6)
    )

    print("🎨 计算完成！正在提取并绘制热力图...")
    
    # 🎯 核心修复：提取最新版的 path_gain 属性
    pg = radio_map.path_gain
    
    # 取第一个基站的数据，并去掉多余的维度，防止 imshow 绘图报错
    pg_np = np.squeeze(pg[0].numpy())
    
    # 将 Path Gain (线性值) 转换为 Path Loss (dB)
    # 加上 1e-10 防止在完全没有信号的盲区出现 log10(0) 报错
    path_loss_db = -10.0 * np.log10(pg_np + 1e-10)

    plt.figure(figsize=(10, 8))
    plt.imshow(path_loss_db, cmap="viridis_r", origin="lower")
    plt.colorbar(label="Path Loss (dB)")
    plt.title("28GHz Radio Map (Paris Etoile)")
    plt.savefig("coverage_map_result.png", dpi=300, bbox_inches='tight')
    
    print("-" * 40)
    print("🎉 恭喜通关！图片已成功生成！")
    print("👉 结果已保存：coverage_map_result.png")
    print("-" * 40)

if __name__ == "__main__":
    main()