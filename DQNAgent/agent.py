# agent_pro.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# ------------------ MODELLO DQN ------------------
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# ------------------ AGENTE DQN ------------------
class DQNAgent:
    def __init__(self, state_size, action_size, device=None, replay_buffer_size=30000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size

        # Hyperparameters
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        self.learning_rate = 0.001

        # Device
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  DQNAgent running on {self.device}")

        # Models
        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # Initialize target model
        self.update_target_model()

    # ------------------ FUNZIONI BASE ------------------
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done,valid_mask, next_valid_mask):
        # Flatten stati se multidimensionali
        state = state.flatten() if len(state.shape) > 1 else state
        next_state = next_state.flatten() if len(next_state.shape) > 1 else next_state
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state, valid_actions=None):
        state = state.flatten() if len(state.shape) > 1 else state

        # Epsilon-greedy
        if np.random.rand() <= self.epsilon:
            if valid_actions is not None:
                return np.random.choice(valid_actions)
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()[0]

        # Maschera azioni non valide
        if valid_actions is not None:
            masked = np.full(self.action_size, -np.inf)
            masked[valid_actions] = q_values[valid_actions]
            return np.argmax(masked)
        return np.argmax(q_values)

    def replay(self, n_steps=1):
        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(n_steps):
            minibatch = random.sample(self.replay_buffer, self.batch_size)

            states = torch.FloatTensor(np.array([x[0] for x in minibatch])).to(self.device)
            actions = torch.LongTensor([x[1] for x in minibatch]).to(self.device)
            rewards = torch.FloatTensor([x[2] for x in minibatch]).to(self.device)
            next_states = torch.FloatTensor(np.array([x[3] for x in minibatch])).to(self.device)
            dones = torch.FloatTensor([x[4] for x in minibatch]).to(self.device)

            # Q-values correnti
            current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Q-values target
            with torch.no_grad():
                next_q = self.target_model(next_states).max(1)[0]
                target_q = rewards + (1 - dones) * self.gamma * next_q

            # Loss e backprop
            loss = self.criterion(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # ------------------ SALVATAGGIO / CARICAMENTO ------------------
    def save(self, filename):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, filename)
        print(f"✓ Modello salvato: {filename}")

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"✓ Modello caricato: {filename}")
