import os
import math
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from isac_env import CellFreeISAC3DEnv

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, device):
        self.max_size = int(max_size)
        self.device = device
        self.ptr = 0
        self.size = 0
        
        self.states = torch.zeros((self.max_size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((self.max_size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((self.max_size, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((self.max_size, state_dim), dtype=torch.float32, device=device)
        self.undones = torch.zeros((self.max_size, 1), dtype=torch.float32, device=device)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        self.undones[self.ptr] = 0.0 if done else 1.0
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        indices = torch.randint(0, self.size, size=(batch_size,), device=self.device)
        return (self.states[indices], self.actions[indices], 
                self.rewards[indices], self.undones[indices], self.next_states[indices])

class ActorSAC(nn.Module):
    def __init__(self, state_dim, action_dim, net_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, net_dim), nn.ReLU(),
            nn.Linear(net_dim, net_dim), nn.ReLU(),
            nn.Linear(net_dim, net_dim), nn.ReLU(),
        )
        self.mean_layer = nn.Linear(net_dim, action_dim)
        self.log_std_layer = nn.Linear(net_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x).clamp(-20, 2)
        return mean, log_std

    def get_action(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        noise = torch.randn_like(mean)
        action = torch.tanh(mean + std * noise) 
        return action

    def get_action_logprob(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        noise = torch.randn_like(mean)
        action_raw = mean + std * noise
        action = torch.tanh(action_raw)
        log_prob = -0.5 * (((action_raw - mean) / std).pow(2) + math.log(2 * math.pi)) - log_std
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=1, keepdim=True)

class CriticTwin(nn.Module):
    def __init__(self, state_dim, action_dim, net_dim):
        super().__init__()
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, net_dim), nn.ReLU(),
            nn.Linear(net_dim, net_dim), nn.ReLU(),
            nn.Linear(net_dim, 1)
        )
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, net_dim), nn.ReLU(),
            nn.Linear(net_dim, net_dim), nn.ReLU(),
            nn.Linear(net_dim, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1_net(sa), self.q2_net(sa)

    def get_q_min(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return torch.min(self.q1_net(sa), self.q2_net(sa))

class AgentModSAC:
    def __init__(self, state_dim, action_dim, net_dim=256, lr=1e-4, device='cpu'):
        self.device = torch.device(device)
        self.gamma = 0.95
        self.tau = 0.005 
        
        self.act = ActorSAC(state_dim, action_dim, net_dim).to(self.device)
        self.cri = CriticTwin(state_dim, action_dim, net_dim).to(self.device)
        self.cri_target = deepcopy(self.cri).to(self.device)
        
        self.act_optimizer = torch.optim.Adam(self.act.parameters(), lr=lr)
        self.cri_optimizer = torch.optim.Adam(self.cri.parameters(), lr=lr / 3.0)
        
        self.alpha = 0.05 
        self.obj_c = 1.0 
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state_ts = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.act.get_action(state_ts)
            return action.cpu().numpy()[0]

    def update_net(self, buffer, batch_size, repeat_times):
        update_times = int(repeat_times) 
        obj_critics = 0.0
        obj_actors = 0.0
        update_a = 0
        
        for update_c in range(1, update_times + 1):
            states, actions, rewards, undones, next_ss = buffer.sample(batch_size)
            
            with torch.no_grad():
                next_as, next_logprobs = self.act.get_action_logprob(next_ss)
                next_qs = self.cri_target.get_q_min(next_ss, next_as)
                q_labels = rewards + undones * self.gamma * (next_qs - next_logprobs * self.alpha)

            q1, q2 = self.cri(states, actions)
            obj_critic = self.criterion(q1, q_labels) + self.criterion(q2, q_labels)
            
            self.cri_optimizer.zero_grad()
            obj_critic.backward()
            torch.nn.utils.clip_grad_norm_(self.cri.parameters(), max_norm=1.0)
            self.cri_optimizer.step()
            
            obj_critics += obj_critic.item()
            
            for param, target_param in zip(self.cri.parameters(), self.cri_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            self.obj_c = 0.995 * self.obj_c + 0.005 * obj_critic.item()
            reliable_lambda = math.exp(-self.obj_c ** 2)

            if update_a / update_c < 1 / (2 - reliable_lambda): 
                update_a += 1
                action_pg, log_prob = self.act.get_action_logprob(states)
                q_value_pg = self.cri.get_q_min(states, action_pg) 
                
                obj_actor = (log_prob * self.alpha - q_value_pg).mean() 
                
                self.act_optimizer.zero_grad()
                obj_actor.backward()
                self.act_optimizer.step()
                obj_actors += obj_actor.item()

        return obj_critics / update_times, (obj_actors / update_a if update_a > 0 else 0)

class EnvWrapper:
    def __init__(self, objective_type):
        self.env = CellFreeISAC3DEnv(M=2, N=2, K=15, objective_type=objective_type)
        self.state_dim = self.env.state_dim 
        self.action_dim = self.env.action_space.shape[0] 
        self.max_step = self.env.max_steps
        self.curr_step = 0

    def reset(self):
        self.curr_step = 0
        return self.env.reset()

    def step(self, continuous_action):
        next_state, reward, done, info = self.env.step(continuous_action)
        self.curr_step += 1
        if self.curr_step >= self.max_step: done = True
        return next_state, reward, done, info

def main():
    print("启动 SAC 智能体训练...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"训练设备: {device}")

    objective = 'max-sum'  
    env = EnvWrapper(objective_type=objective)
    
    agent = AgentModSAC(state_dim=env.state_dim, action_dim=env.action_dim, 
                        net_dim=256,    
                        lr=1e-4,        
                        device=device)
    buffer = ReplayBuffer(max_size=1e5,       
                          state_dim=env.state_dim, 
                          action_dim=env.action_dim, 
                          device=device)
    
    max_episodes = 2000
    batch_size = 256
    start_training_size = 2000 
    
    print("正在进行初始随机探索填充 Buffer...")
    state = env.reset()
    for _ in range(start_training_size):
        action = np.random.uniform(-1, 1, env.action_dim)
        next_state, reward, done, _ = env.step(action)
        buffer.add(state, action, reward, next_state, done=False)
        state = next_state if not done else env.reset()

    global_best_reward = -float('inf')
    global_best_rate = 0.0
    global_best_fim = 0.0
    global_best_coords = ""
    global_best_ep = 0

    drl_best_history = []
    for episode in range(max_episodes):
        state = env.reset()
        ep_reward = 0
        ep_n_rate, ep_n_fim, ep_n_pen, ep_sum_rate, ep_fim = 0.0, 0.0, 0.0, 0.0, 0.0
        actual_steps = 0 
        ep_best_raw_reward = -float('inf')
        for step in range(env.max_step):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            ep_reward += reward 
            buffer.add(state, action, reward, next_state, done=False)
            
            if reward > ep_best_raw_reward:
                ep_best_raw_reward = reward

            ep_n_rate += info.get("norm_rate", 0.0)
            ep_n_fim += info.get("norm_fim", 0.0)
            ep_n_pen += info.get("norm_penalty", 0.0)
            ep_sum_rate += info.get("sum_rate", 0.0)
            ep_fim += info.get("fim_det", 0.0)
            
            if reward > global_best_reward:
                global_best_reward = reward
                global_best_rate = info.get("sum_rate", 0.0)
                global_best_fim = info.get("fim_det", 0.0)
                global_best_ep = episode
                best_indices = info.get('selected_aps', [])
                global_best_coords = " | ".join([
                    f"编号{idx:3d}(X:{env.env.candidate_positions[idx][0]:>5.0f}, "
                    f"Y:{env.env.candidate_positions[idx][1]:>5.0f}, "
                    f"Z:{env.env.candidate_positions[idx][2]:>4.0f})" 
                    for idx in best_indices
                ])
            
            state = next_state
            actual_steps += 1
            if done: break
                
        ep_loss_c, ep_loss_a = 0.0, 0.0
        if buffer.size >= batch_size:
            ep_loss_c, ep_loss_a = agent.update_net(buffer, batch_size, repeat_times=32)

        avg_reward = ep_reward / actual_steps
        avg_n_rate = ep_n_rate / actual_steps
        avg_n_fim = ep_n_fim / actual_steps
        avg_n_pen = ep_n_pen / actual_steps
        avg_sum_rate = ep_sum_rate / actual_steps
        avg_fim = ep_fim / actual_steps
        
        selected_indices = info.get('selected_aps', [])
        coords_str = " | ".join([
            f"编号{idx:3d}(X:{env.env.candidate_positions[idx][0]:>5.0f}, "
            f"Y:{env.env.candidate_positions[idx][1]:>5.0f}, "
            f"Z:{env.env.candidate_positions[idx][2]:>4.0f})" 
            for idx in selected_indices
        ])
        
        print(f" Ep: {episode:3d} | Avg Reward: {avg_reward:>6.2f} | Cri-Loss: {ep_loss_c:>6.1f} | Alpha: 0.05 (Locked)")
        print(f"    回合平均奖励拆解 -> 通信: {avg_n_rate:.2f} | 感知: {avg_n_fim:.2f} | 空间微调: -{avg_n_pen:.2f}")
        print(f"    回合平均物理性能 -> 通信: {avg_sum_rate:.2f} bps/Hz | 感知 FIM: {avg_fim:.2e}")
        print(f"    回合末基站坐标: [ {coords_str} ]\n")
        #pure_physical_reward = avg_n_rate + avg_n_fim
        drl_best_history.append(ep_best_raw_reward)

    print("\n" + "="*60)
    print(f" 训练结束，全局最优解出现于 Ep: {global_best_ep}")
    print(f" 全局最优系统效能 -> 通信 Sum-Rate: {global_best_rate:.2f} bps/Hz | 感知 FIM: {global_best_fim:.2e}")
    print(f" 最优基站部署策略 -> [ {global_best_coords} ]")
    print("="*60 + "\n")
    
    np.save('drl_real_curve.npy', np.array(drl_best_history))
    print(" DRL 训练数据已保存至 'drl_real_curve.npy'")
if __name__ == '__main__':
    main()