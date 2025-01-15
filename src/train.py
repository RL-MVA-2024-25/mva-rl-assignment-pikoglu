import os
import math  # Added for logarithmic calculations
import tqdm
import random
import numpy as np
import torch
import copy  # Replacing "from copy import deepcopy"
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity)  # Buffer capacity
        self.storage = []  # Renamed from 'data' to 'storage' for uniqueness
        self.position = 0  # Renamed from 'index' to 'position'
        self.device = device

    def store(self, state, action, reward, next_state, done):
        if len(self.storage) < self.capacity:
            self.storage.append(None)
        self.storage[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample_batch(self, batch_size):
        batch = random.sample(self.storage, batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.tensor(state, dtype=torch.float32).to(self.device),
            torch.tensor(action, dtype=torch.long).to(self.device),
            torch.tensor(reward, dtype=torch.float32).to(self.device),
            torch.tensor(next_state, dtype=torch.float32).to(self.device),
            torch.tensor(done, dtype=torch.float32).to(self.device),
        )

    def __len__(self):
        return len(self.storage)


def select_action(network, state):
    device = next(network.parameters()).device
    with torch.no_grad():
        q_values = network(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device))
        return torch.argmax(q_values).item()


class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layers=2, activation_fn=torch.nn.ReLU(), norm=None):
        super(NeuralNetwork, self).__init__()
        layer_list = [torch.nn.Linear(input_dim, hidden_dim), activation_fn]
        for _ in range(layers - 1):
            layer_list.extend([torch.nn.Linear(hidden_dim, hidden_dim), activation_fn])
            if norm == 'batch':
                layer_list.append(torch.nn.BatchNorm1d(hidden_dim))
            elif norm == 'layer':
                layer_list.append(torch.nn.LayerNorm(hidden_dim))
        layer_list.append(torch.nn.Linear(hidden_dim, output_dim))
        self.network = torch.nn.Sequential(*layer_list)

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    def __init__(self, config, model):
        self.device = next(model.parameters()).device
        self.action_space_size = config.get('action_space_size', 4)
        self.gamma = config.get('gamma', 0.95)
        self.batch_size = config.get('batch_size', 100)
        buffer_capacity = config.get('buffer_capacity', int(1e5))
        self.memory = ReplayBuffer(buffer_capacity, self.device)
        self.epsilon_max = config.get('epsilon_max', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay_steps = config.get('epsilon_decay_steps', 1000)
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_decay_steps
        self.current_epsilon = self.epsilon_max
        self.policy_net = model
        self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.target_net.eval()
        self.loss_fn = config.get('loss_fn', torch.nn.MSELoss())
        learning_rate = config.get('learning_rate', 0.001)
        self.optimizer = config.get('optimizer', torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate))
        self.grad_steps_per_update = config.get('grad_steps_per_update', 1)
        self.target_update_method = config.get('target_update_method', 'replace')
        self.target_update_frequency = config.get('target_update_frequency', 20)
        self.target_tau = config.get('target_tau', 0.005)
        self.monitor_trials = config.get('monitor_trials', 0)
        self.monitor_interval = config.get('monitor_interval', 10)
        self.save_path = config.get('save_path', './agent.pth')
        self.save_interval = config.get('save_interval', 100)
        self.learn_steps = 0
        self.highest_reward = 0

    def evaluate_policy(self, env, trials):
        total_rewards = []
        discounted_rewards = []
        for _ in range(trials):
            state, _ = env.reset()
            done, truncated = False, False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or truncated):
                action = select_action(self.policy_net, state)
                next_state, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                discounted_reward += (self.gamma ** step) * reward
                state = next_state
                step += 1
            total_rewards.append(total_reward)
            discounted_rewards.append(discounted_reward)
        return np.mean(discounted_rewards), np.mean(total_rewards)

    def initial_state_value(self, env, trials):
        with torch.no_grad():
            values = []
            for _ in range(trials):
                state, _ = env.reset()
                q_vals = self.policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device))
                values.append(q_vals.max().item())
            return np.mean(values)

    def perform_gradient_step(self):
        if len(self.memory) < self.batch_size:
            return  # Not enough data to learn

        states, actions, rewards, next_states, dones = self.memory.sample_batch(self.batch_size)

        # Compute target Q-values
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            targets = rewards + (1 - dones) * self.gamma * max_next_q

        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute loss
        loss = self.loss_fn(current_q, targets)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_steps += 1

    def train(self, env, max_episodes):
        returns = []
        mc_total = []
        mc_discounted = []
        initial_values = []
        episode_count = 0
        cumulative_reward = 0
        state, _ = env.reset()
        step_count = 0

        while episode_count < max_episodes:
            # Epsilon decay
            if step_count > 0 and self.current_epsilon > self.epsilon_min:
                self.current_epsilon -= self.epsilon_step
                self.current_epsilon = max(self.current_epsilon, self.epsilon_min)

            # Action selection
            if random.random() < self.current_epsilon:
                action = env.action_space.sample()
            else:
                action = select_action(self.policy_net, state)

            # Environment step
            next_state, reward, done, truncated, _ = env.step(action)
            self.memory.store(state, action, reward, next_state, done)
            cumulative_reward += reward

            # Gradient steps
            for _ in range(self.grad_steps_per_update):
                self.perform_gradient_step()

            # Target network update
            if self.target_update_method == 'replace':
                if step_count % self.target_update_frequency == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            elif self.target_update_method == 'soft':
                target_dict = self.target_net.state_dict()
                policy_dict = self.policy_net.state_dict()
                for key in policy_dict:
                    target_dict[key] = self.target_tau * policy_dict[key] + (1 - self.target_tau) * target_dict[key]
                self.target_net.load_state_dict(target_dict)

            step_count += 1

            # Episode termination
            if done or truncated:
                episode_count += 1

                # Convert cumulative reward to power-of-10 format
                if cumulative_reward > 0:
                    ep_return_exp = math.log10(cumulative_reward)
                else:
                    ep_return_exp = 0  # Handle non-positive rewards

                # Monitoring
                if self.monitor_trials > 0 and episode_count % self.monitor_interval == 0:
                    mc_dr, mc_tr = self.evaluate_policy(env, self.monitor_trials)
                    v0 = self.initial_state_value(env, self.monitor_trials)
                    mc_total.append(mc_tr)
                    mc_discounted.append(mc_dr)
                    initial_values.append(v0)
                    returns.append(cumulative_reward)

                    print(
                        f"Episode {episode_count:2d}, Epsilon {self.current_epsilon:6.2f}, "
                        f"Memory Size {len(self.memory):4d}, "
                        f"Ep Return=10^{ep_return_exp:.2f}, MC Total {mc_tr:6.0f}, "
                        f"MC Discounted {mc_dr:6.0f}, V0 {v0:6.0f}"
                    )

                    if mc_tr > self.highest_reward:
                        self.highest_reward = mc_tr
                        self.save(self.save_path)
                        print(f"New highest return achieved: {self.highest_reward}")
                else:
                    returns.append(cumulative_reward)
                    print(
                        f"Episode {episode_count:2d}, Epsilon {self.current_epsilon:6.2f}, "
                        f"Memory Size {len(self.memory):4d}, Ep Return=10^{ep_return_exp:.2f}"
                    )

                # Reset for next episode
                state, _ = env.reset()
                cumulative_reward = 0
            else:
                state = next_state

        return returns, mc_discounted, mc_total, initial_values

    def act(self, state):
        return select_action(self.policy_net, state)

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Agent model saved to {path}")

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.policy_net.eval()
        print(f"Agent model loaded from {path}")


# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the neural network model
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = NeuralNetwork(
    input_dim=state_dim,
    hidden_dim=512,
    output_dim=action_dim,
    layers=5,
    activation_fn=torch.nn.SiLU(),
    norm=None
).to(device)

# Configuration dictionary
config = {
    'action_space_size': action_dim,
    'learning_rate': 0.001,
    'gamma': 0.98,
    'buffer_capacity': 1000000,
    'epsilon_min': 0.01,
    'epsilon_max': 1.0,
    'epsilon_decay_steps': 10000,
    'grad_steps_per_update': 2,
    'target_update_method': 'soft',  # 'replace' or 'soft'
    'target_update_frequency': 600,  # Relevant if method is 'replace'
    'target_tau': 0.001,  # Relevant if method is 'soft'
    'batch_size': 1024,
    'loss_fn': torch.nn.SmoothL1Loss(),
    'monitor_trials': 50,
    'monitor_interval': 50,
    'save_path': './dqn_agent.pth',
    'save_interval': 50
}

# Instantiate the agent
agent = DQNAgent(config, model)


class ProjectAgent:
    def __init__(self):
        self.dqn_agent = DQNAgent(config, copy.deepcopy(model))  # Ensure separate instances

    def act(self, observation, use_random=False):
        return self.dqn_agent.act(observation)

    def save(self, path):
        self.dqn_agent.save(path)

    def load(self):
        default_path = os.path.join(os.getcwd(), "dqn_agent.pth")
        self.dqn_agent.load(default_path)


def populate_replay_buffer(env, agent, desired_size):
    state, _ = env.reset()
    progress = tqdm.tqdm(total=desired_size, desc="Populating Replay Buffer")
    for _ in range(desired_size):
        action = agent.act(state)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.dqn_agent.memory.store(state, action, reward, next_state, done)
        if done or truncated:
            state, _ = env.reset()
        else:
            state = next_state
        progress.update(1)
    progress.close()


if __name__ == "__main__":
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # env.seed(seed)  # Uncomment if the environment supports seeding

    # Initialize the ProjectAgent
    project_agent = ProjectAgent()

    # Populate the replay buffer
    populate_replay_buffer(env, project_agent, desired_size=8000)

    # Train the agent
    episode_returns, discounted_rewards, total_rewards, initial_vals = project_agent.dqn_agent.train(env, max_episodes=4000)

    # Save the trained agent
    project_agent.save(config['save_path'])
    print("Training completed successfully.")

