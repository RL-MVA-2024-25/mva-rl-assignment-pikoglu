import os
import random
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

# Initialize the environment with a step limit of 200
env = TimeLimit(env=HIVPatient(domain_randomization=False), max_episode_steps=200)

# Determine the computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define state and action dimensions based on the environment
state_dimension = env.observation_space.shape[0]
action_space_size = env.action_space.n

# Configuration settings for the agent
configuration = {
    "nb_actions": action_space_size,
    "learning_rate": 1e-3,
    "gamma": 0.95,
    "buffer_size": 1_000_000,
    "epsilon_min": 1e-2,
    "epsilon_max": 1.0,
    "epsilon_decay_steps": 10_000,
    "epsilon_initial_delay": 400,
    "batch_size": 500,
    "gradient_steps_per_update": 2,
    "target_update_method": "ema",  # Options: 'ema' or 'replace'
    "target_update_interval": 600,
    "target_update_tau": 1e-3,
    "loss_function": torch.nn.SmoothL1Loss(),
    "evaluation_trials": 50
}

class ExperienceReplay:
    def __init__(self, capacity, device):
        self.capacity = int(capacity)
        self.memory = []
        self.current_index = 0
        self.device = device

    def store(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.current_index] = (state, action, reward, next_state, done)
        self.current_index = (self.current_index + 1) % self.capacity

    def sample_batch(self, batch_size):
        sampled = random.sample(self.memory, batch_size)
        transposed = list(zip(*sampled))
        return [torch.tensor(np.array(items), dtype=torch.float32).to(self.device) for items in transposed]

    def __len__(self):
        return len(self.memory)

def select_greedy_action(model, state):
    model_device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        q_values = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(model_device))
        return torch.argmax(q_values).item()

class DeepQNetwork(nn.Module):
    def __init__(self, state_dim=state_dimension, action_dim=action_space_size, layers=4, hidden_units=512):
        super(DeepQNetwork, self).__init__()
        network_layers = [nn.Linear(state_dim, hidden_units)]
        for _ in range(layers - 1):
            network_layers.append(nn.Linear(hidden_units, hidden_units))
        self.hidden_layers = nn.ModuleList(network_layers)
        self.output_layer = nn.Linear(hidden_units, action_dim)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)

class ProjectAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DeepQNetwork().to(self.device)
        self.action_count = configuration["nb_actions"]
        self.discount_factor = configuration.get("gamma", 0.95)
        self.batch_size = configuration.get("batch_size", 100)
        buffer_capacity = configuration.get("buffer_size", int(1e5))
        self.replay_memory = ExperienceReplay(buffer_capacity, self.device)
        self.epsilon_max = configuration.get("epsilon_max", 1.0)
        self.epsilon_min = configuration.get("epsilon_min", 1e-2)
        self.epsilon_decay_steps = configuration.get("epsilon_decay_steps", 10_000)
        self.epsilon_initial_delay = configuration.get("epsilon_initial_delay", 20)
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_decay_steps
        self.target_net = deepcopy(self.policy_net).to(self.device)
        self.loss_fn = configuration.get("loss_function", nn.MSELoss())
        learning_rate = configuration.get("learning_rate", 1e-3)
        self.optimizer = configuration.get("optimizer", torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate))
        self.grad_steps = configuration.get("gradient_steps_per_update", 1)
        self.target_method = configuration.get("target_update_method", "replace")
        self.target_freq = configuration.get("target_update_interval", 20)
        self.target_tau = configuration.get("target_update_tau", 5e-3)
        self.eval_trials = configuration.get("evaluation_trials", 0)

    def act(self, state):
        return select_greedy_action(self.policy_net, state)

    def save(self, filepath):
        torch.save(self.policy_net.state_dict(), filepath)

    def load(self):
        checkpoint_path = os.path.join(os.getcwd(), "last_DQN.pt")
        self.policy_net.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.policy_net.eval()

    def evaluate_performance(self, environment, trials):
        total_rewards = []
        discounted_rewards = []
        for _ in range(trials):
            state, _ = environment.reset()
            done = False
            truncated = False
            cumulative_reward = 0
            discounted_reward = 0
            step_count = 0
            while not (done or truncated):
                action = select_greedy_action(self.policy_net, state)
                next_state, reward, done, truncated, _ = environment.step(action)
                state = next_state
                cumulative_reward += reward
                discounted_reward += (self.discount_factor ** step_count) * reward
                step_count += 1
            total_rewards.append(cumulative_reward)
            discounted_rewards.append(discounted_reward)
        return np.mean(discounted_rewards), np.mean(total_rewards)

    def estimate_initial_value(self, environment, trials):
        with torch.no_grad():
            values = []
            for _ in range(trials):
                state, _ = environment.reset()
                q_values = self.policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device))
                values.append(q_values.max().item())
        return np.mean(values)

    def perform_gradient_update(self):
        if len(self.replay_memory) > self.batch_size:
            states, actions, rewards, next_states, dones = self.replay_memory.sample_batch(self.batch_size)
            with torch.no_grad():
                max_next_q = self.target_net(next_states).max(dim=1)[0]
                targets = rewards + self.discount_factor * max_next_q * (1 - dones.squeeze())
            current_q = self.policy_net(states).gather(1, actions.long().unsqueeze(1))
            loss = self.loss_fn(current_q, targets.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, environment, max_episodes):
        returns = []
        avg_disc_rewards = []
        avg_total_rewards = []
        initial_values = []
        episode_count = 0
        cumulative_reward = 0
        current_state, _ = environment.reset()
        epsilon = self.epsilon_max
        total_steps = 0
        best_score = 0

        while episode_count < max_episodes:
            # Decay epsilon after the initial delay
            if total_steps > self.epsilon_initial_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = environment.action_space.sample()
            else:
                action = select_greedy_action(self.policy_net, current_state)
            
            # Execute action
            next_state, reward, done, truncated, _ = environment.step(action)
            self.replay_memory.store(current_state, action, reward, next_state, done)
            cumulative_reward += reward

            # Perform training steps
            for _ in range(self.grad_steps):
                self.perform_gradient_update()
            
            # Update target network if required
            if self.target_method == "replace":
                if total_steps % self.target_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            elif self.target_method == "ema":
                target_dict = self.target_net.state_dict()
                policy_dict = self.policy_net.state_dict()
                for key in policy_dict:
                    target_dict[key] = self.target_tau * policy_dict[key] + (1 - self.target_tau) * target_dict[key]
                self.target_net.load_state_dict(target_dict)
            
            total_steps += 1

            # Check if the episode has ended
            if done or truncated:
                episode_count += 1

                # Evaluation and monitoring
                if self.eval_trials > 0 and episode_count % 200 == 0:
                    avg_disc, avg_tot = self.evaluate_performance(environment, self.eval_trials)
                    initial_val = self.estimate_initial_value(environment, self.eval_trials)
                    avg_total_rewards.append(avg_tot)
                    avg_disc_rewards.append(avg_disc)
                    initial_values.append(initial_val)
                    returns.append(cumulative_reward)
                    print(f"Episode {episode_count:2d}, Epsilon {epsilon:.2e}, Buffer Size {len(self.replay_memory):4d}, "
                          f"Ep Return {cumulative_reward:.1e}, MC Total {avg_tot:.2e}, MC Discounted {avg_disc:.2e}, V0 {initial_val:.2e}")

                    if avg_tot > best_score:
                        best_score = avg_tot
                        print(f"New best score: {best_score:.2e}")
                        torch.save(self.policy_net.state_dict(), "DQN")
                        print("Best model saved.")
                else:
                    returns.append(cumulative_reward)
                    print(f"Episode {episode_count:2d}, Epsilon {epsilon:.2e}, Buffer Size {len(self.replay_memory):4d}, "
                          f"Ep Return {cumulative_reward:.1e}")

                # Reset for the next episode
                current_state, _ = environment.reset()
                cumulative_reward = 0
            else:
                current_state = next_state

        return returns, avg_disc_rewards, avg_total_rewards, initial_values

def set_seed(seed_value: int = 42):
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



