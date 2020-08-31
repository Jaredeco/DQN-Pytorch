import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
import random
from torch import optim
from tqdm import tqdm


class DQN(nn.Module):
    def __init__(self, n_actions, n_inputs, lr=0.01):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_inputs, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, n_actions)

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
    def __init__(self, gamma=0.99, epsilon=1):
        self.env = gym.make('CartPole-v0')
        self.n_actions = self.env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon
        self.policy_net = DQN(self.n_actions, 4)
        self.target_net = DQN(self.n_actions, 4)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.replay_memory = collections.deque(maxlen=10000)
        self.min_replay_memory_size = 100
        self.batch_size = 64
        self.scores = []
        self.eps_min = 0.05
        self.eps_dec = 5e-4
        self.iter_count = 0
        self.update_target = 10

    def update_replay_memory(self, obs):
        self.replay_memory.append(obs)

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            q = self.policy_net(torch.tensor([state], dtype=torch.float32).cuda())
            action = torch.argmax(q).item()
        else:
            action = self.env.action_space.sample()
        return action

    def train(self):
        if len(self.replay_memory) < self.batch_size:
            return
        self.policy_net.optimizer.zero_grad()
        batch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, dones = [x[0] for x in batch], [x[1] for x in batch], [x[2] for x in batch], [x[3] for x in batch], [x[4] for x in batch]
        state_batch = torch.tensor(states, dtype=torch.float32).cuda()
        next_state_batch = torch.tensor(next_states, dtype=torch.float32).cuda()
        action_batch = actions
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        reward_batch = torch.tensor(rewards).cuda()
        done_batch = torch.tensor(dones).cuda()
        current_qs_list = self.policy_net(
            state_batch)[batch_index, action_batch]
        future_qs_list = self.target_net(next_state_batch)
        future_qs_list[done_batch] = 0.0
        new_q = reward_batch + self.gamma * torch.max(future_qs_list, dim=1)[0]
        loss = self.policy_net.loss(current_qs_list, torch.as_tensor(new_q, dtype=torch.float32).cuda())
        loss.backward()
        self.policy_net.optimizer.step()

    def step(self):
        done = False
        state = self.env.reset()
        score = 0
        while not done:
            action = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            score += reward
            self.update_replay_memory((state, action, reward, next_state, done))
            if not self.iter_count % self.update_target:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
            state = next_state
        self.scores.append(score)


agent = Agent()
episodes = 900
render_from = int(episodes * 0.95)
for episode in tqdm(range(episodes)):
    agent.step()
    agent.train()
    if episode > render_from:
        agent.env.render()
avg_score = np.mean(agent.scores)
print(f"Average score {int(avg_score)}")
agent.env.close()
