import numpy as np
import math
import gym
from gym import spaces
import os
import sys

def get_resource_path_env(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

class CellFreeISAC3DEnv(gym.Env):
    def __init__(self, M=2, N=2, K=15, objective_type='max-sum'):
        super(CellFreeISAC3DEnv, self).__init__()
        self.objective_type = objective_type 
        print(f"正在初始化 3D Cell-Free ISAC 环境...")
        
        self.pg_database_5d = np.load(get_resource_path_env('paris_pg_database_3d.npy'))
        self.pg_database = self.pg_database_5d.reshape(-1, 125, 125)      
        self.candidate_positions = np.load(get_resource_path_env('candidate_positions_3d.npy')) 
        
        self.blockage_threshold_db = -110.0 
        self.blockage_threshold_linear = 10 ** (self.blockage_threshold_db / 10.0)
        
        zero_pixel_ratio = np.sum(self.pg_database < self.blockage_threshold_linear, axis=(1, 2)) / (125 * 125)
        self.valid_indices = np.where(zero_pixel_ratio <= 0.80)[0]
        self.num_candidates = len(self.valid_indices) 
        
        self.M = M
        self.N = N
        self.K = K
        self.x_min, self.x_max = -250.0, 250.0
        self.y_min, self.y_max = -250.0, 250.0
        self.grid_resolution = 4.0 
        
        np.random.seed(2026) 
        self.users_info = [] 
        avg_coverage_map = np.mean(self.pg_database, axis=0) 
        hotspots = [
            {"mean": [-100, -100], "cov": [[900, 0], [0, 900]], "weight": 0.4}, 
            {"mean": [150, 50],   "cov": [[400, 0], [0, 400]], "weight": 0.4}, 
            {"mean": [0, 200],    "cov": [[625, 0], [0, 625]], "weight": 0.2}  
        ]
        
        while len(self.users_info) < self.K:
            rnd = np.random.rand()
            cumulative_weight = 0
            for hs in hotspots:
                cumulative_weight += hs["weight"]
                if rnd <= cumulative_weight:
                    selected_hotspot = hs
                    break
            test_x, test_y = np.random.multivariate_normal(selected_hotspot["mean"], selected_hotspot["cov"])
            if not (self.x_min <= test_x <= self.x_max and self.y_min <= test_y <= self.y_max):
                continue
            row, col = self._coord_to_grid(test_x, test_y)
            if avg_coverage_map[row, col] > self.blockage_threshold_linear:
                user_z = np.random.uniform(1.0, 2.0) 
                is_indoor = False
            else:
                user_z = np.random.uniform(5.0, 30.0) 
                is_indoor = True
            self.users_info.append({"pos": [test_x, test_y, user_z], "is_indoor": is_indoor})
        
        self.num_targets = 2
        self.targets_params = [
            {"type": "circle", "center": [-100, -100], "radius": 80.0, "speed": 0.05, "z": 40.0, "phase": 0.0},
            {"type": "line", "start": [200, 200], "end": [-200, -200], "speed": 0.02, "z": 20.0, "phase": 0.0} 
        ]
        self.targets_pos = []
        self.time_step = 0
        self.max_steps = 200 

        self.state_dim = (self.K * 3) + (self.num_targets * 2) 
        self.observation_space = spaces.Box(low=-300.0, high=300.0, shape=(self.state_dim,), dtype=np.float32)
        
        # 🎯 回归 12 维虚拟坐标系统！
        self.action_dim = (self.M + self.N) * 3
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)

    def reset(self):
        self.time_step = 0
        self.targets_pos = []
        for p in self.targets_params:
            p["phase"] = np.random.uniform(0, 2 * math.pi)
            if p["type"] == "circle":
                x = p["center"][0] + p["radius"] * math.cos(p["phase"])
                y = p["center"][1] + p["radius"] * math.sin(p["phase"])
            elif p["type"] == "line":
                x, y = p["start"][0], p["start"][1]
            self.targets_pos.append([x, y, p["z"]])
        return self.get_state()

    def get_state(self):
        state = []
        for u_info in self.users_info:
            pos = u_info["pos"]
            state.extend([pos[0]/250.0, pos[1]/250.0, pos[2]/50.0]) 
        for pos in self.targets_pos:
            state.extend([pos[0]/250.0, pos[1]/250.0]) 
        return np.array(state, dtype=np.float32)

    def _update_target_pos(self):
        self.time_step += 1
        for i, p in enumerate(self.targets_params):
            if p["type"] == "circle":
                current_angle = p["phase"] + p["speed"] * self.time_step
                new_x = p["center"][0] + p["radius"] * math.cos(current_angle)
                new_y = p["center"][1] + p["radius"] * math.sin(current_angle)
                self.targets_pos[i] = [new_x, new_y, p["z"]]
            elif p["type"] == "line":
                progress = (math.sin(p["phase"] + p["speed"] * self.time_step) + 1.0) / 2.0
                new_x = p["start"][0] + progress * (p["end"][0] - p["start"][0])
                new_y = p["start"][1] + progress * (p["end"][1] - p["start"][1])
                self.targets_pos[i] = [new_x, new_y, p["z"]]

    def _coord_to_grid(self, x, y):
        col = int(np.clip((x - self.x_min) / self.grid_resolution, 0, 124))
        row = int(np.clip((y - self.y_min) / self.grid_resolution, 0, 124))
        return row, col

    def _get_beta_lk(self, ap_idx, target_x, target_y, is_indoor=False, target_z=None):
        if target_z is not None:
            ap_pos = self.candidate_positions[ap_idx]
            d = max(math.sqrt((target_x - ap_pos[0])**2 + (target_y - ap_pos[1])**2 + (target_z - ap_pos[2])**2), 1.0)
            c, f = 3e8, 28e9
            gain_linear = (c / (4 * math.pi * d * f)) ** 2
            return gain_linear

        row, col = self._coord_to_grid(target_x, target_y)
        gain_linear = self.pg_database[ap_idx, row, col]
        if gain_linear < self.blockage_threshold_linear or np.isnan(gain_linear):
            return 0.0 
        if is_indoor:
            penetration_loss_db = np.random.uniform(20.0, 35.0)
            gain_linear = gain_linear * (10 ** (-penetration_loss_db / 10.0))
        return gain_linear

    def calculate_communication_rate(self, selected_ap_indices):
        sum_rate = 0.0
        min_rate = float('inf')
        for k in range(self.K):
            u_info = self.users_info[k]
            uk_x, uk_y, _ = u_info["pos"]
            is_indoor = u_info["is_indoor"]
            snr_k = 0.0
            for ap_idx in selected_ap_indices:
                snr_k += self._get_beta_lk(ap_idx, uk_x, uk_y, is_indoor)
            rate_k = math.log2(1 + snr_k * 1e12)
            sum_rate += rate_k
            if rate_k < min_rate: min_rate = rate_k
        return sum_rate, min_rate

    def _calculate_single_fim_det(self, tx_indices, rx_indices, target_pos):
        tp_x, tp_y, tp_z = target_pos
        F11, F22, F33, F12, F13, F23 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for m_idx in tx_indices:
            beta_m = self._get_beta_lk(m_idx, tp_x, tp_y, target_z=tp_z)
            if beta_m == 0.0: continue
            tm_pos = self.candidate_positions[m_idx]
            d_m = max(math.sqrt((tp_x - tm_pos[0])**2 + (tp_y - tm_pos[1])**2 + (tp_z - tm_pos[2])**2), 1e-3)
            ux_m, uy_m, uz_m = (tp_x - tm_pos[0])/d_m, (tp_y - tm_pos[1])/d_m, (tp_z - tm_pos[2])/d_m
            
            for n_idx in rx_indices:
                beta_n = self._get_beta_lk(n_idx, tp_x, tp_y, target_z=tp_z)
                if beta_n == 0.0: continue
                rn_pos = self.candidate_positions[n_idx]
                d_n = max(math.sqrt((tp_x - rn_pos[0])**2 + (tp_y - rn_pos[1])**2 + (tp_z - rn_pos[2])**2), 1e-3)
                ux_n, uy_n, uz_n = (tp_x - rn_pos[0])/d_n, (tp_y - rn_pos[1])/d_n, (tp_z - rn_pos[2])/d_n
                
                weight = beta_m * beta_n * 1e20 
                sum_x, sum_y, sum_z = ux_m + ux_n, uy_m + uy_n, uz_m + uz_n
                F11 += weight * (sum_x ** 2)
                F22 += weight * (sum_y ** 2)
                F33 += weight * (sum_z ** 2)
                F12 += weight * (sum_x * sum_y)
                F13 += weight * (sum_x * sum_z)
                F23 += weight * (sum_y * sum_z)
                
        fim_det = F11 * (F22 * F33 - F23 ** 2) - F12 * (F12 * F33 - F13 * F23) + F13 * (F12 * F23 - F13 * F22)
        return max(fim_det, 1e-12)

    def calculate_sensing_fim(self, tx_indices, rx_indices):
        total_fim_det = 0.0
        for t_pos in self.targets_pos:
            total_fim_det += self._calculate_single_fim_det(tx_indices, rx_indices, t_pos)
        return total_fim_det

    def step(self, continuous_action):
        num_aps = self.M + self.N
        
        # 1. 从 Actor 获取并格式化连续虚拟坐标 (范围 -1 到 1)
        raw_action = np.clip(continuous_action, -1.0, 1.0)
        virtual_norm = raw_action.reshape((num_aps, 3))
        
        # 2. 映射到物理空间进行吸附
        virtual_phys = np.zeros_like(virtual_norm)
        virtual_phys[:, 0] = virtual_norm[:, 0] * 250.0  
        virtual_phys[:, 1] = virtual_norm[:, 1] * 250.0  
        virtual_phys[:, 2] = virtual_norm[:, 2] * 15.0 + 20.0 
        
        valid_positions = self.candidate_positions[self.valid_indices]
        top_k_indices = []
        snapped_phys = []
        
        # 吸附到最近物理基站
        for i in range(num_aps):
            dist_sq = np.sum((valid_positions - virtual_phys[i])**2, axis=1)
            for already_selected_idx in top_k_indices:
                idx_in_valid = np.where(self.valid_indices == already_selected_idx)[0][0]
                dist_sq[idx_in_valid] = np.inf
            best_idx_in_valid = np.argmin(dist_sq)
            real_idx = self.valid_indices[best_idx_in_valid]
            top_k_indices.append(real_idx)
            snapped_phys.append(valid_positions[best_idx_in_valid])
            
        top_k_indices_arr = np.array(top_k_indices)
        tx_indices = top_k_indices_arr[:self.M]
        rx_indices = top_k_indices_arr[self.M:]
        
        sum_rate, min_rate = self.calculate_communication_rate(top_k_indices_arr)
        fim_det = self.calculate_sensing_fim(tx_indices, rx_indices)
        
        # =========================================================================
        # 🎯 核心护城河：计算虚拟坐标与被吸附坐标的归一化 MSE 误差！
        # 让神经网络能够顺着这个“引力漏斗”完美地贴合物理基站！
        # =========================================================================
        snapped_phys = np.array(snapped_phys)
        snapped_norm = np.zeros_like(snapped_phys)
        # 将被选中的物理基站反向缩放到 [-1, 1] 空间，用于和 Actor 输出做对比
        snapped_norm[:, 0] = snapped_phys[:, 0] / 250.0
        snapped_norm[:, 1] = snapped_phys[:, 1] / 250.0
        snapped_norm[:, 2] = (snapped_phys[:, 2] - 20.0) / 15.0
        
        # 计算完美平滑的连续梯度惩罚（乘以 3.0 放大梯度引导）
        norm_penalty = np.sum((virtual_norm - snapped_norm)**2) * 3.0 
        
        # 通信得分与感知得分
        safe_sum_rate = max(sum_rate, 1e-5) 
        norm_rate = safe_sum_rate / 10.0
        safe_fim_det = max(fim_det, 1e-15) 
        norm_fim = (math.log10(safe_fim_det) + 12.0) / 2.0
        
        if self.objective_type == 'max-sum':
            reward = norm_rate + norm_fim - norm_penalty
        elif self.objective_type == 'max-min':
            norm_min_rate = max(min_rate, 1e-5) / 2.0 
            reward = norm_min_rate + norm_fim - norm_penalty
        else:
            reward = 0.0
            
        self._update_target_pos()
        next_state = self.get_state()
        done = self.time_step >= self.max_steps
        
        info = {
            "selected_aps": top_k_indices_arr, 
            "sum_rate": sum_rate, "min_rate": min_rate, "fim_det": fim_det,
            "norm_rate": norm_rate, "norm_fim": norm_fim, "norm_penalty": norm_penalty
        }
        return next_state, reward, done, info