# train.py

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from env_hiv import HIVPatient

# 1. History Wrapper
class HistoryWrapper(gym.Wrapper):
    def __init__(self, env, history_length=4):
        super(HistoryWrapper, self).__init__(env)
        self.history_length = history_length
        self.state_history = deque(maxlen=history_length)
        self.action_history = deque(maxlen=history_length)
        
        original_space = env.observation_space
        action_space = env.action_space
        
        if isinstance(action_space, gym.spaces.Discrete):
            action_dim = 1  # Represent actions as integers
        else:
            action_dim = action_space.shape[0]
        
        new_low = np.concatenate([original_space.low for _ in range(history_length)] +
                                 [np.array([0]) for _ in range(history_length)])
        new_high = np.concatenate([original_space.high for _ in range(history_length)] +
                                  [np.array([env.action_space.n - 1]) for _ in range(history_length)])
        
        self.observation_space = gym.spaces.Box(low=new_low, high=new_high, dtype=np.float32)
    
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        for _ in range(self.history_length):
            self.state_history.append(observation)
            self.action_history.append(0)  # No-action placeholder
        return self._get_observation(), info
    
    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        self.state_history.append(observation)
        self.action_history.append(action)
        return self._get_observation(), reward, done, truncated, info
    
    def _get_observation(self):
        state_history = np.concatenate(list(self.state_history))
        action_history = np.array(list(self.action_history)).reshape(-1, 1)
        # Normalize states with log(1 + obs)
        normalized_states = np.log1p(state_history)
        # Combine normalized states and action history
        return np.concatenate([normalized_states, action_history.flatten()]).astype(np.float32)

# 2. Deep Dueling DQN Network
class DeepDuelingDQN(nn.Module):
    def __init__(self, state_dim=28, action_dim=4):  # Adjusted state_dim based on history_length=4
        super(DeepDuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 256),   # Feature Layer 1
            nn.ReLU(),
            nn.Linear(256, 256),         # Feature Layer 2
            nn.ReLU(),
            nn.Linear(256, 256),         # Feature Layer 3
            nn.ReLU(),
            nn.Linear(256, 256),         # Feature Layer 4
            nn.ReLU(),
            nn.Linear(256, 256),         # Feature Layer 5
            nn.ReLU(),
            nn.Linear(256, 256),         # Feature Layer 6
            nn.ReLU()
        )
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_vals

# 3. Prioritized Replay Buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity=200000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32),
                indices,
                weights)
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio
    
    def __len__(self):
        return len(self.buffer)

# 4. ProjectAgent with Enhanced Architecture and GPU Utilization
class ProjectAgent:
    def __init__(self):
        # Hyperparameters
        self.gamma = 0.99
        self.lr = 1e-3
        self.batch_size = 64
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        history_length = 4
        original_state_dim = 6
        action_dim = 4
        state_dim = (original_state_dim + 1) * history_length  # e.g., 6 states + 1 action per history step
        
        # Initialize networks
        self.q_network = DeepDuelingDQN(state_dim, action_dim).to(self.device)
        self.target_network = DeepDuelingDQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        # Replay Buffer
        self.memory = PrioritizedReplayBuffer(capacity=200000)
        self.update_counter = 0
        self.update_target_steps = 200  # Frequency to update target network
        self.beta_start = 0.4
        self.beta_frames = 100000
        self.frame = 1  # Frame counter for PER

        # Track best average reward for checkpointing
        self.best_avg_reward = -float('inf')
    
    def act(self, observation, use_random=False):
        """
        Implement epsilon-greedy: with probability epsilon, choose a random action;
        otherwise, choose argmax Q(s,a).
        """
        if use_random and random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
            action = q_values.argmax(dim=1).item()
            return action
    
    def update(self):
        """
        Sample from replay buffer and update Q-network.
        """
        if len(self.memory) < self.batch_size:
            return

        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size, beta=beta)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        weights_t = torch.FloatTensor(weights).to(self.device).unsqueeze(1)

        # Current Q(s,a)
        q_values = self.q_network(states_t).gather(1, actions_t)

        # Target Q(s,a)
        with torch.no_grad():
            best_actions = self.q_network(next_states_t).argmax(dim=1, keepdim=True)
            target_q = self.target_network(next_states_t).gather(1, best_actions)
            y = rewards_t + self.gamma * target_q * (1 - dones_t)

        # Compute TD error
        td_errors = y - q_values
        loss = (weights_t * nn.MSELoss(reduction='none')(q_values, y)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update priorities
        new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, new_priorities.flatten())

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.update_target_steps == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.frame += 1
    
    def save(self, path):
        torch.save(self.q_network.state_dict(), path)
    
    def load(self):
        model_path = "best_dqn.pt"
        self.q_network.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.q_network.to(self.device)
        self.target_network.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.target_network.to(self.device)

# 5. Training Routine
if __name__ == "__main__":
    base_env = HIVPatient(domain_randomization=False)
    wrapped_env = HistoryWrapper(base_env, history_length=4)  # Adjust history_length as needed
    env = TimeLimit(wrapped_env, max_episode_steps=200)
    agent = ProjectAgent()

    NUM_EPISODES = 300  # Train for ~300 episodes (tweak as needed)
    for episode in range(NUM_EPISODES):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0

        while not done and not truncated:
            action = agent.act(obs, use_random=True)  # Use epsilon-greedy
            next_obs, reward, done, truncated, _ = env.step(action)
            agent.memory.push(obs, action, reward, next_obs, done or truncated)

            obs = next_obs
            episode_reward += reward

            agent.update()
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}, Epsilon: {agent.epsilon:.3f}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")
            if avg_reward > agent.best_avg_reward:
                agent.best_avg_reward = avg_reward
                agent.save("best_dqn.pt")
                print(f"New best average reward: {agent.best_avg_reward:.2f}. Model saved as best_dqn.pt.")

    # After training, save the final model
    agent.save("my_dqn.pt")
    print("Training complete. Models saved as my_dqn.pt and best_dqn.pt.")
