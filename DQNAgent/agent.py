import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Rete neurale con Dueling Architecture
class DQN(nn.Module):
    def __init__(self, state_size, action_size, dueling=False):
        super(DQN, self).__init__()
        self.dueling = dueling
        
        # Layers condivisi (più neuroni per gestire mappe complesse)
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        
        if dueling:
            # Dueling architecture
            self.fc_value = nn.Linear(256, 128)
            self.value = nn.Linear(128, 1)
            
            self.fc_adv = nn.Linear(256, 128)
            self.advantage = nn.Linear(128, action_size)
        else:
            self.fc3 = nn.Linear(256, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        if self.dueling:
            # Value stream
            v = torch.relu(self.fc_value(x))
            v = self.value(v)
            
            # Advantage stream
            a = torch.relu(self.fc_adv(x))
            a = self.advantage(a)
            
            # Combine: Q = V + (A - mean(A))
            q = v + a - a.mean(dim=1, keepdim=True)
            return q
        else:
            return self.fc3(x)

# DQN Agent migliorato
class DQNAgent:
    def __init__(self, state_size, action_size, device=None, replay_buffer_size=100000, batch_size=64, dueling=True, prioritized=False):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        self.prioritized = prioritized
        
        # Per prioritized experience replay
        if prioritized:
            self.priorities = deque(maxlen=replay_buffer_size)

        # Hyperparameters 
        self.gamma = 0.99  # Fattore di sconto base (verrà modificato esternamente per le 16x16)
        self.epsilon = 1.0 
        self.epsilon_min = 0.05 
        self.epsilon_decay = 0.9995 
        self.learning_rate = 0.00025 
        self.target_update_freq = 1000 
        self.tau = 0.005 # Soft update
        self.step_count = 0 
        
        self.use_double_dqn = True

        # Device
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQNAgent running on {self.device}")

        # Models
        self.model = DQN(state_size, action_size, dueling=dueling).to(self.device)
        self.target_model = DQN(state_size, action_size, dueling=dueling).to(self.device)
        
        # --- OTTIMIZZATORE ADAMW ---
        # Weight Decay: Previene overfitting su traiettorie specifiche
        # Amsgrad: Migliora la convergenza in ambienti stocastici
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4, amsgrad=True)
        
        # Huber Loss è più robusta agli outlier rispetto a MSE
        self.criterion = nn.HuberLoss() 

        self.update_target_model(hard=True) # Primo sync hard

    def update_target_model(self, hard=False):
        if hard:
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            # Soft update
            for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done, valid_mask, next_valid_mask):
        state = state.flatten() if len(state.shape) > 1 else state
        next_state = next_state.flatten() if len(next_state.shape) > 1 else next_state
        
        self.replay_buffer.append((state, action, reward, next_state, done))
        if self.prioritized:
            self.priorities.append(max(self.priorities, default=1.0))

    def act(self, state, valid_actions=None):
        state = state.flatten() if len(state.shape) > 1 else state

        # Epsilon-greedy
        if np.random.rand() <= self.epsilon:
            if valid_actions is not None:
                return random.choice(valid_actions)
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()[0]
        self.model.train()

        # Maschera azioni non valide
        if valid_actions is not None:
            masked = np.full(self.action_size, -np.inf)
            masked[valid_actions] = q_values[valid_actions]
            return np.argmax(masked)
        return np.argmax(q_values)

    def _sample_prioritized_batch(self):
        if len(self.priorities) == 0:
            return random.sample(range(len(self.replay_buffer)), self.batch_size)
        
        probs = np.array(self.priorities)
        probs = probs / probs.sum()
        indices = np.random.choice(len(self.replay_buffer), self.batch_size, p=probs)
        self._last_minibatch_indices = indices
        return indices

    def replay(self, n_steps=1):
        if len(self.replay_buffer) < self.batch_size:
            return None

        total_loss = 0
        
        for _ in range(n_steps):
            if self.prioritized:
                minibatch_indices = self._sample_prioritized_batch()
                minibatch = [self.replay_buffer[i] for i in minibatch_indices]
            else:
                minibatch = random.sample(self.replay_buffer, self.batch_size)

            states = torch.FloatTensor(np.array([x[0] for x in minibatch])).to(self.device)
            actions = torch.LongTensor([x[1] for x in minibatch]).to(self.device)
            rewards = torch.FloatTensor([x[2] for x in minibatch]).to(self.device)
            next_states = torch.FloatTensor(np.array([x[3] for x in minibatch])).to(self.device)
            dones = torch.FloatTensor([x[4] for x in minibatch]).to(self.device)

            # Q-values correnti
            current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Double DQN
            with torch.no_grad():
                if self.use_double_dqn:
                    next_actions = self.model(next_states).argmax(1)
                    next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                else:
                    next_q = self.target_model(next_states).max(1)[0]
                
                target_q = rewards + (1 - dones) * self.gamma * next_q

            # Loss e ottimizzazione
            loss = self.criterion(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # Clipping gradienti
            self.optimizer.step()
            
            total_loss += loss.item()

            # Update priorità
            if self.prioritized and hasattr(self, '_last_minibatch_indices'):
                td_errors = (target_q - current_q).detach().cpu().numpy()
                for idx, td_error in zip(self._last_minibatch_indices, td_errors):
                    if idx < len(self.priorities):
                        self.priorities[idx] = abs(td_error) + 1e-6

            self.step_count += 1
            if self.step_count % self.target_update_freq == 0:
                self.update_target_model()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return total_loss / n_steps

    def save(self, filename):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, filename)
        # print(f"Modello salvato: {filename}")

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Modello caricato: {filename}")