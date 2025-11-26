# agent.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state','action','reward','next_state','done'))

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_size, action_size, device=None,
                 lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 target_update=10):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.policy_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 64
        self.target_update = target_update
        self.steps_done = 0

    def act(self, state_np, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        state = torch.from_numpy(state_np).float().to(self.device)
        with torch.no_grad():
            q = self.policy_net(state).cpu().numpy()[0]
        masked = np.full_like(q, -np.inf)
        masked[valid_actions] = q[valid_actions]
        return int(masked.argmax())

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state.astype(np.float32), action, reward, next_state.astype(np.float32), done)

    def replay(self, n_steps=1):
        if len(self.replay_buffer) < self.batch_size:
            return
        for _ in range(n_steps):
            transitions = self.replay_buffer.sample(self.batch_size)
            states = torch.from_numpy(np.vstack(transitions.state)).float().to(self.device)
            actions = torch.tensor(transitions.action, dtype=torch.long, device=self.device).unsqueeze(1)
            rewards = torch.tensor(transitions.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
            next_states = torch.from_numpy(np.vstack(transitions.next_state)).float().to(self.device)
            dones = torch.tensor(transitions.done, dtype=torch.float32, device=self.device).unsqueeze(1)


            q_values = self.policy_net(states).gather(1, actions)


            with torch.no_grad():
                q_next_all = self.target_net(next_states)
                q_next_max, _ = q_next_all.max(dim=1, keepdim=True)

            q_target = rewards + (1.0 - dones) * (self.gamma * q_next_max)

            loss = nn.functional.mse_loss(q_values, q_target)

            self.optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
            self.optimizer.step()

            self.steps_done += 1

        # soft/periodic update target network
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
