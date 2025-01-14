# train.py

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from gymnasium.wrappers import TimeLimit

from env_hiv import HIVPatient

class DQNNetwork(nn.Module):
    def __init__(self, state_dim=6, action_dim=4):
        super(DQNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

class ProjectAgent:
    def __init__(self):
        self.gamma = 0.99
        self.lr = 1e-3
        self.batch_size = 64
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DQNNetwork().to(self.device)
        self.target_network = DQNNetwork().to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.memory = ReplayBuffer()
        self.update_counter = 0
        self.update_target_steps = 200  # How often to update target net

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

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device).unsqueeze(1)

        # Current Q(s,a)
        q_values = self.q_network(states_t).gather(1, actions_t)

        # Target Q(s,a)
        with torch.no_grad():
            best_actions = self.q_network(next_states_t).argmax(dim=1, keepdim=True)
            target_q = self.target_network(next_states_t).gather(1, best_actions)
            y = rewards_t + self.gamma * target_q * (1 - dones_t)

        loss = nn.MSELoss()(q_values, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.update_target_steps == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save(self.q_network.state_dict(), path)

    def load(self):
        model_path = "my_dqn.pt"
        self.q_network.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.q_network.to(self.device)
        self.target_network.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.target_network.to(self.device)

if __name__ == "__main__":
    env = TimeLimit(HIVPatient(domain_randomization=False), max_episode_steps=200)
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

            # Train (update) after each step
            agent.update()

        if (episode+1) % 10 == 0:
            print(f"Episode {episode+1}, Epsilon: {agent.epsilon:.3f}, Reward: {episode_reward:.2f}")

    agent.save("my_dqn.pt")
    print("Training complete. Model saved as my_dqn.pt.")
