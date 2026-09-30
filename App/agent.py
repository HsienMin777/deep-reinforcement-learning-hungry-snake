"""DQN network, replay buffer, and agent."""
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import (BATCH_SIZE, DEVICE, GAMMA, LR, MEMORY_SIZE, N_ACTIONS,
                    STATE_DIM)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class DQNNet(nn.Module):
    def __init__(self, input_dim=STATE_DIM, output_dim=N_ACTIONS):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity=MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class Agent:
    """一條蛇對應一個 Agent：各自擁有 policy net、target net、optimizer 與 replay buffer。"""

    def __init__(self, input_dim=STATE_DIM):
        self.policy_net = DQNNet(input_dim).to(DEVICE)
        self.target_net = DQNNet(input_dim).to(DEVICE)
        self.sync_target()
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer()

    def select_action(self, state, eps):
        if random.random() < eps:
            return random.randrange(N_ACTIONS)
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(state_t).argmax(dim=1).item()

    def sync_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = self.memory.sample(BATCH_SIZE)

        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=DEVICE)
        action_batch = torch.tensor(batch.action, dtype=torch.int64, device=DEVICE).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=DEVICE)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        # Q(s, a)
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Target Q = r + gamma * max_a' Q_target(s', a')，終止狀態不 bootstrap
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            expected_q_values = reward_batch + (1 - done_batch) * GAMMA * next_q_values

        loss = F.mse_loss(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # 避免梯度爆炸
        self.optimizer.step()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        """載入權重；失敗時丟出例外，由呼叫端決定如何處理。"""
        state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
        self.policy_net.load_state_dict(state_dict)
        self.sync_target()
