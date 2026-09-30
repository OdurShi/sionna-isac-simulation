import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # 强制 CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import sionna
from sionna.rt import load_scene, Transmitter, PlanarArray, RadioMapSolver
import time

def main():
    # 1. 加载场景与核心物理参数配置
    scene = load_scene(sionna.rt.scene.etoile)
    scene.frequency = 28e9 
    
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

    # 2. 定义三维候选区域
    x_coords = np.linspace(-250, 250, 11)
    y_coords = np.linspace(-250, 250, 11)
    
    # ✅ 核心修改 1：增加 Z 轴的高度分层采样 (例如从 5米 到 35米，步长 5米)
    z_coords = np.linspace(5, 35, 7) 
    
    solver = RadioMapSolver()
    solver.max_depth = 4 # 默认通常是 3，增加一次反射计算
    # ✅ 核心修改 2：预先分配一个 5D 的 NumPy 数组，方便 SAC 算法后期直接通过 3D 索引查表
    # 维度含义: [X坐标索引, Y坐标索引, Z坐标索引, 接收机X网格数(125), 接收机Y网格数(125)]
    database_3d = np.zeros((len(x_coords), len(y_coords), len(z_coords), 125, 125), dtype=np.float32)
    positions = []

    total_points = len(x_coords) * len(y_coords) * len(z_coords)
    print(f"🚀 开始生成 3D 离线指纹库，共计 {total_points} 个候选点...")
    start_time = time.time()

    count = 0
    # 嵌套循环：遍历 3D 空间中的每一个候选基站位置
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            for k, z in enumerate(z_coords):
                count += 1
                pos = [float(x), float(y), float(z)]
                positions.append(pos)
                
                # 添加临时全向基站
                tx = Transmitter(name="temp_tx", position=pos, orientation=[0, 0, 0])
                scene.add(tx)
                
                # 射线追踪计算
                # 注意：接收机(用户)所在的覆盖面始终固定在 z=1.5 米的地面高度
                radio_map = solver(scene, 
                                   cell_size=[4, 4], 
                                   center=[0, 0, 1.5], 
                                   size=[500, 500],
                                   orientation=[0, 0, 0],
                                   samples_per_tx=int(5e6)) 
                
                # 获取线性信道增益 (Linear Path Gain)
                pg = radio_map.path_gain.numpy().astype(np.float32)
                
                # 直接将这张二维热力图存入 5D 矩阵对应的 3D 槽位中
                database_3d[i, j, k, :, :] = np.squeeze(pg)
                
                # 移除当前基站，为下一个点腾出空间
                scene.remove("temp_tx")
                
                # 打印进度 (因为计算量变大了，可以适当调整打印频率)
                if count % 20 == 0:
                    elapsed = time.time() - start_time
                    print(f"✅ 已完成 {count}/{total_points} | 耗时: {elapsed:.1f}s")

    # 3. 保存数据库
    np.save('paris_pg_database_3d.npy', database_3d)
    np.save('candidate_positions_3d.npy', np.array(positions))
    
    print("-" * 50)
    print(f"🎉 3D 数据库生成完毕！总耗时: {(time.time()-start_time)/60:.1f} 分钟")
    print(f"👉 数据库矩阵最终形状: {database_3d.shape}")
    print("👉 paris_pg_database_3d.npy (线性信道增益矩阵)")
    print("👉 candidate_positions_3d.npy (对应的3D坐标)")
    print("-" * 50)

if __name__ == "__main__":
    main()