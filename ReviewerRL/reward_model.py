import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativeRewardModel(nn.Module):
    def __init__(self, state_size, action_size, dropout_rate=0.1):
        super(RelativeRewardModel, self).__init__()
        
        # Input: 2 action + state size
        self.input_dim = state_size + 2 * action_size
        
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.dropout1 = nn.Dropout(dropout_rate)
        

        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(128, 64)
        self.ln3 = nn.LayerNorm(64)
        self.dropout3 = nn.Dropout(dropout_rate * 0.5)
        
        # Output: 1 logit (1 for action1 > action2, -1 for action2 > action1)
        self.output = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, state, action1_one_hot, action2_one_hot):
        # Concatenate state and both actions
        x = torch.cat([state, action1_one_hot, action2_one_hot], dim=-1)
        
        x = torch.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = torch.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = torch.relu(self.ln3(self.fc3(x)))
        x = self.dropout3(x)
        
        return self.output(x)

    def predict_preference(self, state, action1_one_hot, action2_one_hot):
        self.eval()
        with torch.no_grad():
            # Get logit for action1 > action2
            logits = self.forward(state, action1_one_hot, action2_one_hot)
            prob = self.sigmoid(logits)
        self.train()
        return prob
    
    def rank_actions(self, state, action_list):
        self.eval()
        with torch.no_grad():
            scores = {}
            
            # Compare each action against all others
            for i, (idx_a, action_a) in enumerate(action_list):
                total_score = 0.0
                comparisons = 0
                
                for j, (idx_b, action_b) in enumerate(action_list):
                    if i == j:
                        continue
                    
                    # Predict preference of action_a over action_b
                    prob = self.predict_preference(state, action_a, action_b)
                    total_score += prob.item()
                    comparisons += 1
                
                # Average score for action a with respect to all others
                avg_score = total_score / comparisons if comparisons > 0 else 0.5
                scores[idx_a] = avg_score
            
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        self.train()
        return ranked


class RewardShaper:
    @staticmethod
    def distance_reward(prev_dist, current_dist, goal_reached=False):
        if goal_reached:
            return 1.0
        
        if prev_dist is None or current_dist is None:
            return 0.0
        
        # Normalize reward in [-0.5, 0.5]
        delta = prev_dist - current_dist
        max_delta = 5.0  
        normalized_reward = max(min(delta / max_delta, 0.5), -0.5)
        
        return normalized_reward
    
    @staticmethod
    def action_efficiency_reward(action, optimal_action):
        if action == optimal_action:
            return 1.0
        
        if action in ["LEFT", "RIGHT"] and optimal_action in ["LEFT", "RIGHT"]:
            return 0.5
        
        # if action is valid but not optimal, give small positive reward
        return 0.1
    
    @staticmethod
    def safety_penalty(info):
        penalty = 0.0
        
        if info.get("wall_bump_count", 0) > 0:
            penalty -= 0.5
        if info.get("lava_death", False):
            penalty -= 1.0
        if info.get("forward_blocked", False):
            penalty -= 0.3
        if info.get("timeout", False):
            penalty -= 0.8
        if info.get("spin_count", 0) > 3:
            penalty -= 0.4
        
        return max(penalty, -1.0) 
    
    @staticmethod
    def combine_rewards(distance_r, efficiency_r, safety_r, weights=(0.4, 0.4, 0.2)):
        w_dist, w_eff, w_safe = weights
        return w_dist * distance_r + w_eff * efficiency_r + w_safe * safety_r


class SequenceRewardModel(nn.Module):
    def __init__(self, state_size, action_size, max_sequence_length=5, dropout_rate=0.1):
        super(SequenceRewardModel, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.max_seq_len = max_sequence_length
        
        # Encoder for the state (shared across the sequence)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Encoder for individual actions
        self.action_encoder = nn.Sequential(
            nn.Linear(action_size, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # LSTM to process the temporal sequence of actions
        # Input: action embedding (64) concatenated with state (128) = 192
        self.lstm_input_size = 128 + 64
        self.lstm_hidden_size = 128
        self.sequence_lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate if dropout_rate > 0 else 0
        )
        
        # Output heads
        self.action_value_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 1)
        )
        
        self.sequence_value_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 1)
        )
        
        # Coherence head to evaluate how well the actions in the sequence are coordinated
        self.coherence_head = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state, action_sequence, sequence_mask=None):
        batch_size = state.size(0)
        
        # state: (batch, state_size)
        state_features = self.state_encoder(state)
        
        # state_features: (batch, 128) -> expand to (batch, max_seq_len, 128) for concatenation
        state_features_expanded = state_features.unsqueeze(1).expand(
            -1, self.max_seq_len, -1
        )
        
        # Encode each action in the sequence
        # action_sequence: (batch, max_seq_len, action_size)
        action_sequence_flat = action_sequence.view(-1, self.action_size)  
        action_features_flat = self.action_encoder(action_sequence_flat) 
        action_features = action_features_flat.view(batch_size, self.max_seq_len, -1) 
        
        # Concatenate state features with action features for LSTM input
        lstm_input = torch.cat([state_features_expanded, action_features], dim=-1) 
        
        # Pass through LSTM
        lstm_output, (hidden, cell) = self.sequence_lstm(lstm_input)
        
        # Apply mask
        if sequence_mask is not None:
            # sequence_mask: (batch, max_seq_len) -> (batch, max_seq_len, 1)
            lstm_output = lstm_output * sequence_mask.unsqueeze(-1)
        
        action_scores = self.action_value_head(lstm_output).squeeze(-1)
        

        final_hidden = hidden[-1]  
        sequence_score = self.sequence_value_head(final_hidden)
        
        coherence = self.coherence_head(final_hidden) 
        
        return {
            'sequence_score': sequence_score,
            'action_scores': action_scores,
            'coherence': coherence,
            'hidden_states': lstm_output
        }
    
    def compare_sequences(self, state, sequence1, sequence2, mask1=None, mask2=None):
        out1 = self.forward(state, sequence1, mask1)
        out2 = self.forward(state, sequence2, mask2)
        
        # Combine score and coherence into a single preference score
        score1 = out1['sequence_score'] + 0.3 * (out1['coherence'] - 0.5) * 2  
        score2 = out2['sequence_score'] + 0.3 * (out2['coherence'] - 0.5) * 2
        
        logits = score1 - score2
        
        return logits
    
    def rank_sequences(self, state, sequence_list, mask_list=None):
        self.eval()
        with torch.no_grad():
            scores = []
            
            for i, seq in enumerate(sequence_list):
                # Prepare input
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(state.device)  # (1, max_seq_len, action_size)
                mask_tensor = None
                if mask_list is not None and mask_list[i] is not None:
                    mask_tensor = torch.FloatTensor(mask_list[i]).unsqueeze(0).to(state.device)
                
                out = self.forward(state, seq_tensor, mask_tensor)
                
                # Combined score
                combined_score = out['sequence_score'].item() + 0.3 * (out['coherence'].item() - 0.5) * 2
                scores.append((i, combined_score))
            
            # Sort by descending score
            ranked = sorted(scores, key=lambda x: x[1], reverse=True)
        
        self.train()
        return ranked
    
    def get_feedback(self, state, action_sequence, sequence_mask=None):
        self.eval()
        with torch.no_grad():
            # Prepare input
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
            seq_tensor = torch.FloatTensor(action_sequence).unsqueeze(0).to(next(self.parameters()).device)
            mask_tensor = None
            if sequence_mask is not None:
                mask_tensor = torch.FloatTensor(sequence_mask).unsqueeze(0).to(next(self.parameters()).device)
            
            out = self.forward(state_tensor, seq_tensor, mask_tensor)
            
            overall_score = out['sequence_score'].item()
            action_scores = out['action_scores'].squeeze(0).cpu().numpy().tolist()
            coherence = out['coherence'].item()
            
            suggestions = []
            
            # Check coherence
            if coherence < 0.5:
                suggestions.append("Sequence is not coherent: consider more consistent actions")
            elif coherence > 0.8:
                suggestions.append("Sequence is well-coordinated")
            
            # Check action scores
            if sequence_mask is not None:
                valid_actions = int(sequence_mask.sum())
            else:
                valid_actions = len(action_scores)
            
            low_score_actions = [i for i in range(valid_actions) if action_scores[i] < -0.3]
            if low_score_actions:
                suggestions.append(f"Problematic actions: {low_score_actions}")
            
            if overall_score > 0.5:
                suggestions.append("Overall good sequence")
            elif overall_score < -0.3:
                suggestions.append("Problematic sequence, consider alternatives")
        
        self.train()
        
        return {
            'overall_score': overall_score,
            'action_scores': action_scores,
            'coherence': coherence,
            'suggestions': suggestions
        }