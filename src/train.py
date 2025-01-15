from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import tqdm 
import random
import numpy as np
from copy import deepcopy
import os

env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) 
        self.data = []
        self.index = 0 
        self.device = device

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))

    def __len__(self):
        return len(self.data)


def greedy_action(network, state):
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth=2, activation=torch.nn.SiLU(), normalization=None):
        super(MLP, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 1)])
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        if activation is not None:
            self.activation = activation
        else:
            self.activation = torch.nn.ReLU()
        if normalization == 'batch':
            self.normalization = torch.nn.BatchNorm1d(hidden_dim)
        elif normalization == 'layer':
            self.normalization = torch.nn.LayerNorm(hidden_dim)
        else:
            self.normalization = None

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            if self.normalization is not None:
                x = self.normalization(x)
        return self.output_layer(x)


class DQN_Agent:
    def __init__(self, config, model):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.device = device  # Added device attribute
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size, device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop
        self.model = model 
        self.target_model = deepcopy(self.model).to(device)
        self.target_model.eval()  
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0
        self.monitor_every = config['monitor_every'] if 'monitor_every' in config.keys() else 10
        self.save_path = config['save_path'] if 'save_path' in config.keys() else './agent.pth'
        self.save_every = config['save_every'] if 'save_every' in config.keys() else 100
        self.learn_step_counter = 0  # Initialize learn step counter
        self.best_return = 0  # Track best return

    def MC_eval(self, env, nb_trials):
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x, _ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = greedy_action(self.model, x)
                y, r, done, trunc, _ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma ** step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)
    
    def V_initial_state(self, env, nb_trials):
        with torch.no_grad():
            val = []
            for _ in range(nb_trials):
                x, _ = env.reset()
                Q_values = self.model(torch.Tensor(x).unsqueeze(0).to(self.device)).max().item()
                val.append(Q_values)
        return np.mean(val)
    
    def gradient_step(self):
        if len(self.memory) < self.batch_size:
            return  # Not enough samples to learn

        X, A, R, Y, D = self.memory.sample(self.batch_size)
        QYmax = self.target_model(Y).max(1)[0].detach()
        update = torch.addcmul(R, 1 - D, QYmax, value=self.gamma)
        QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
        loss = self.criterion(QXA, update.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        self.learn_step_counter += 1

    def train(self, env, max_episode):
        episode_return = []
        MC_avg_total_reward = []
        MC_avg_discounted_reward = []
        V_init_state = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            # Update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)
            # Select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action(self.model, state)
            # Step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # Train
            for _ in range(self.nb_gradient_steps): 
                self.gradient_step()
            # Update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau * model_state_dict[key] + (1 - tau) * target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)
            # Next transition
            step += 1
            if done or trunc:
                episode += 1
                # Monitoring
                if self.monitoring_nb_trials > 0 and episode % self.monitor_every == 0: 
                    MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)
                    V0 = self.V_initial_state(env, self.monitoring_nb_trials)
                    MC_avg_total_reward.append(MC_tr)
                    MC_avg_discounted_reward.append(MC_dr)
                    V_init_state.append(V0)
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", memory size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ={:e} ", '{:6.0f}'.format(episode_cum_reward), 
                          ", MC tot ", '{:6.0f}'.format(MC_tr),
                          ", MC disc ", '{:6.0f}'.format(MC_dr),
                          ", V0 ", '{:6.0f}'.format(V0),
                          sep='')
                    if MC_tr > self.best_return:
                        self.best_return = MC_tr
                        self.save(self.save_path)
                        print("Best return is updated to ", self.best_return)
                else:
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", memory size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ={:e}", '{:6.0f}'.format(episode_cum_reward), 
                          sep='')
                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state

    def act(self, state):
        return greedy_action(self.model, state)

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {path}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set the model
state_dim = env.observation_space.shape[0]
nb_actions = env.action_space.n

model = MLP(state_dim, 512, nb_actions, depth=5, activation=torch.nn.SiLU(), normalization=None).to(device)

config = {
    'nb_actions': nb_actions,
    'learning_rate': 0.001,
    'gamma': 0.98,
    'buffer_size': 1000000,
    'epsilon_min': 0.01,
    'epsilon_max': 1.,
    'epsilon_decay_period': 10000,
    'epsilon_delay_decay': 400,
    'batch_size': 1024,  # Reduced from 1024 to 128
    'gradient_steps': 2,
    'update_target_strategy': 'ema', # or 'replace'
    'update_target_freq': 600,
    'update_target_tau': 0.001,
    'criterion': torch.nn.SmoothL1Loss(),
    'monitoring_nb_trials': 50, 
    'monitor_every': 50, 
    'save_path': './dqn_agent.pth',
    'save_every': 50
}

agent = DQN_Agent(config, model)

class ProjectAgent:
    def __init__(self):
        self.dqn_agent = DQN_Agent(config, model)
    
    def act(self, observation, use_random=False):
        return self.dqn_agent.act(observation)
    
    def save(self, path):
        self.dqn_agent.save(path)
    
    def load(self):
        path = os.path.join(os.getcwd(), "dqn_agent.pth")  # Corrected path
        self.dqn_agent.load(path)

def fill_buffer(env, agent, buffer_size):
    state, _ = env.reset()
    progress_bar = tqdm.tqdm(total=buffer_size, desc="Filling the replay buffer")
    for _ in range(buffer_size):
        action = agent.act(state)
        next_state, reward, done, trunc, _ = env.step(action)
        agent.dqn_agent.memory.append(state, action, reward, next_state, done)
        if done or trunc:
            state, _ = env.reset()
        else:
            state = next_state
        progress_bar.update(1)
    progress_bar.close()

if __name__ == "__main__":
    # Set the seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # env.seed(seed)  # Uncomment if env supports it
    
    # Initialize ProjectAgent
    project_agent = ProjectAgent()
    
    # Fill the buffer
    fill_buffer(env, project_agent, 8000)
    
    # Train the agent
    ep_length, disc_rewards, tot_rewards, V0 = project_agent.dqn_agent.train(env, 4000)
    project_agent.save(config['save_path'])
    print("Training done")
