import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from torch import optim


class DQN(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 30, 5)
        self.conv3 = nn.Conv2d(30, 40, 5)
        self.fc1 = nn.Linear(40*6*6, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_actions)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters())
        self.cuda()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent:
    def __init__(self, input__dim, n_actions):
        self.n_actions = 6
        self.gamma = 0.98
        self.epsilon = 1
        self.in_dim = 8
        self.policy_net = DQN(input__dim, n_actions)
        self.target_net = DQN(input__dim, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.replay_memory = deque(maxlen=100000)
        self.batch_size = 64
        self.target_net_update_int = 5
        self.train_counter = 0
        self.eps_min = 0.05
        self.eps_dec = 5e-6

    def update_replay_memory(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.random() > self.epsilon:
            with torch.no_grad():
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
        states, actions, rewards, next_states, dones = [x[0] for x in batch], [x[1] for x in batch], \
                                                       [x[2] for x in batch], [x[3] for x in batch], [x[4] for x in batch]
        state_batch = torch.tensor(states, dtype=torch.float32).cuda()
        next_state_batch = torch.tensor(next_states, dtype=torch.float32).cuda()
        done_batch = torch.tensor(dones).cuda()
        reward_batch = torch.tensor(rewards).cuda()
        action_batch = actions
        batch_index = np.arange(self.batch_size)
        current_qs_list = self.policy_net(
            state_batch)[batch_index, action_batch]
        future_qs_list = self.target_net(next_state_batch)
        future_qs_list[done_batch] = 0.0
        new_q = reward_batch + self.gamma * torch.max(future_qs_list, dim=1)[0]
        loss = self.policy_net.loss(current_qs_list, torch.as_tensor(new_q, dtype=torch.float32).cuda())
        loss.backward()
        self.policy_net.optimizer.step()
        self.train_counter += 1

    def update_target_network(self):
        if not self.train_counter % self.target_net_update_int:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decrease_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        torch.save(self.policy_net.state_dict(), "policy_net.pth")
        torch.save(self.target_net.state_dict(), "target_net.pth")

    def load_models(self):
        self.policy_net.load_state_dict(torch.load("policy_net.pth"))
        self.target_net.load_state_dict(torch.load("target_net.pth"))
