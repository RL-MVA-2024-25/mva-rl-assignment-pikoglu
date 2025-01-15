import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import os
from env_hiv import HIVPatient

# Define a reward scaling factor
REWARD_SCALE = 1e6

# 1. Revised Dueling DQN Architecture with Soft Updates
class DuelingDQN(nn.Module):
    """Dueling DQN architecture with Double DQN implementation"""
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        
        # Feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Value stream
        self.value_net = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Advantage stream
        self.advantage_net = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
    def forward(self, x):
        features = self.feature_net(x)
        value = self.value_net(features)
        advantage = self.advantage_net(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

# 2. Standard Replay Buffer
class ReplayBuffer:
    """Simple replay buffer"""
    def __init__(self, capacity=100000):
        self.buffer = []
        self.capacity = capacity
        self.pos = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), 
                np.array(action), 
                np.array(reward, dtype=np.float32),
                np.array(next_state),
                np.array(done, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)

# 3. Agent Class with Enhanced Training Stability
class ProjectAgent:
    """DQN agent that interfaces with the HIV environment"""
    def __init__(self):
        # Training parameters
        self.gamma = 0.99
        self.lr = 1e-5  # Further reduced learning rate
        self.batch_size = 128
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99  # Slower decay rate
        
        # Soft update parameter
        self.tau = 0.001  # For soft target updates
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Network dimensions
        state_dim = 6  # Align with evaluation environment
        self.action_dim = 4  # HIV environment has 4 actions
        
        # Initialize networks
        self.q_net = DuelingDQN(state_dim, self.action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(capacity=100000)
        
        # Initialize counters
        self.update_counter = 0
        self.target_update_freq = 1000  # Not used with soft updates but kept for reference
        
        # Initialize loss function
        self.loss_fn = nn.SmoothL1Loss()  # Huber Loss
    
    def soft_update(self):
        """Soft update model parameters."""
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
    def act(self, state, use_random=False):
        """Select action using epsilon-greedy policy"""
        if use_random and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_tensor)
            action = q_values.argmax(dim=1).item()
            return action
    
    def update(self):
        """Perform one step of training"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Scale rewards
        rewards = rewards / REWARD_SCALE
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values using Double DQN
        with torch.no_grad():
            # Action selection is from the current network
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            # Action evaluation is from the target network
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
        
        # Compute loss using Huber Loss
        loss = self.loss_fn(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)  # Already 1.0
        self.optimizer.step()
        
        # Soft update target network
        self.soft_update()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, path="dqn_model.pt"):
        """Save the model"""
        torch.save({
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path="dqn_model.pt"):
        """Load the model"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}")

# 4. Evaluation Function (Unchanged)
def evaluate_agent(agent, eval_env, num_episodes=5):
    """Evaluate the agent's performance"""
    eval_rewards = []
    
    for episode in range(num_episodes):
        eval_state, _ = eval_env.reset()
        eval_episode_reward = 0
        eval_done = False
        eval_truncated = False
        
        while not eval_done and not eval_truncated:
            eval_action = agent.act(eval_state, use_random=False)  # No exploration
            eval_state, eval_reward, eval_done, eval_truncated, _ = eval_env.step(eval_action)
            eval_episode_reward += eval_reward
        
        eval_rewards.append(eval_episode_reward)
    
    return sum(eval_rewards) / len(eval_rewards)

# 5. Training Routine with Enhanced Logging
def main():
    # Create environments without HistoryWrapper
    base_env = HIVPatient(domain_randomization=False)
    env = TimeLimit(base_env, max_episode_steps=200)
    
    eval_base_env = HIVPatient(domain_randomization=False)
    eval_env = TimeLimit(eval_base_env, max_episode_steps=200)
    
    # Create agent
    agent = ProjectAgent()
    
    # Optionally load a pre-trained model
    # agent.load("best_model.pt")
    
    # Training parameters
    num_episodes = 300  # You can increase this number for better performance
    eval_freq = 10
    best_eval_reward = float('-inf')
    
    print("Starting training...")
    
    try:
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_loss = 0
            episode_steps = 0
            done = False
            truncated = False
            
            while not done and not truncated:
                # Select and perform action
                action = agent.act(state, use_random=True)
                next_state, reward, done, truncated, _ = env.step(action)
                
                # Store transition with scaled reward
                agent.memory.push(state, action, reward, next_state, done or truncated)
                
                # Move to next state
                state = next_state
                episode_reward += reward
                
                # Perform optimization step
                loss = agent.update()
                if loss is not None:
                    episode_loss += loss
                
                episode_steps += 1
            
            # Calculate average loss for the episode
            avg_loss = episode_loss / episode_steps if episode_steps > 0 else 0
            
            # Print episode statistics
            print(f"\nEpisode {episode + 1}")
            print(f"Training reward: {episode_reward:.2f}")
            print(f"Average loss: {avg_loss:.5f}")
            print(f"Epsilon: {agent.epsilon:.3f}")
            
            # Evaluate every eval_freq episodes
            if (episode + 1) % eval_freq == 0:
                avg_eval_reward = evaluate_agent(agent, eval_env)
                print(f"Average evaluation reward: {avg_eval_reward:.2f}")
                
                # Save best model
                if avg_eval_reward > best_eval_reward:
                    best_eval_reward = avg_eval_reward
                    agent.save("dqn_model.pt")
                    print(f"New best model saved with reward: {best_eval_reward:.2f}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    print("\nTraining complete!")
    print(f"Best evaluation reward: {best_eval_reward:.2f}")

if __name__ == "__main__":
    main()

