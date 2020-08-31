import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
import random
from torch import optim
import gym


class DQN(nn.Module):
    def __init__(self, n_inputs, n_actions, lr=0.01):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = 'cuda:0'
        self.loss = nn.MSELoss()
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions


class Agent:
    def __init__(self, n_actions, gamma=0.99, epsilon=1):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.in_dim = 2
        self.policy_net = DQN(self.in_dim, self.n_actions)
        self.target_net = DQN(self.in_dim, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.replay_memory = collections.deque(maxlen=100000)
        self.batch_size = 64
        self.target_update_counter = 0
        self.scores = []
        self.eps_min = 0.01
        self.eps_dec = 5e-5
        self.iter_count = 0
        self.update_target = 50

    def update_replay_memory(self, obs):
        self.replay_memory.append(obs)

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            q = self.policy_net(torch.tensor([state], dtype=torch.float32).cuda())
            action = torch.argmax(q).item()
        else:
            action = random.randint(0, self.n_actions - 1)
        return action

    def train(self):
        if len(self.replay_memory) < self.batch_size:
            return
        self.policy_net.optimizer.zero_grad()
        batch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, dones = [x[0] for x in batch], [x[1] for x in batch], [x[2] for x in batch], [x[3] for x in batch], [x[4] for x in batch]
        state_batch = torch.tensor(states, dtype=torch.float32).cuda()
        next_state_batch = torch.tensor(next_states, dtype=torch.float32).cuda()
        reward_batch = torch.tensor(rewards).cuda()
        action_batch = actions
        done_batch = torch.tensor(dones).cuda()
        batch_index = np.arange(self.batch_size)
        current_qs_list = self.policy_net(state_batch)[batch_index, action_batch]
        future_qs_list = self.target_net(next_state_batch)
        future_qs_list[done_batch] = 0.0
        new_q = reward_batch + self.gamma * torch.max(future_qs_list, dim=1)[0]
        loss = self.policy_net.loss(current_qs_list, torch.as_tensor(new_q, dtype=torch.float32).cuda())
        loss.backward()
        self.policy_net.optimizer.step()
        if not self.iter_count % self.update_target:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self):
        torch.save(self.policy_net.state_dict(), 'policy_net.pth')


# env = gym.make("LunarLander-v2")
env = gym.make("MountainCar-v0")
agent = Agent(env.action_space.n)
print(env.observation_space)
num_episodes = 500
rewards = []
for episode in range(num_episodes):
    done = False
    state = env.reset()
    score = 0
    while not done:
        if episode > 50:
            env.render()
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        score += reward
        agent.update_replay_memory((state, action, reward, next_state, done))
        agent.epsilon = agent.epsilon - agent.eps_dec if agent.epsilon > agent.eps_min else agent.eps_min
        agent.train()
        state = next_state
    rewards.append(score)
    print(f"Episode {episode} Score {int(score)} Epsilon {agent.epsilon}")
print(f"Average score {np.mean(rewards[-100:])}")
