import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import collections
import random


# Agent网络
class Agent_net(nn.Module):
    def __init__(self, args):
        super(Agent_net, self).__init__()
        self.input_size, self.hidden_size, self.output_size = args
        self.fc1 = nn.Linear(self.input_size, 128)
        self.gru_net = nn.GRU(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, hidden_state):
        fc1_op = self.fc1(inputs)

    def get_action(self, inputs, hidden_state):




# 记忆库
class Replaybuffer():
    def __init__(self, args):
        self.size = args
        self.mem_list = collections.deque(maxlen=self.size)

    def save_trans(self, trans):
        self.mem_list.append(trans)

    def sample_batch(self,  batch_size):
        s_ls, a_ls, r_ls, s_next_ls, done_ls = [], [], [], [], []
        trans_batch = random.sample(self.mem_list, batch_size)
        for trans in trans_batch:
            s, a, r, s_next, done = trans
            s_ls.append(s)
            a_ls.append([a])
            r_ls.append([r])
            s_next_ls.append(s_next)
            done_ls.append([done])
        return torch.FloatTensor(s_ls),\
                torch.LongTensor(a_ls),\
                torch.FloatTensor(r_ls),\
                torch.FloatTensor(s_next_ls),\
                torch.FloatTensor(done_ls)
    



# Mix网络
class Mixing_net(nn.Module):  # s 为输入生成正权值
    def __init__(self, args):
        super(Mixing_net, self).__init__()
        self.num_agent_net, self.input_size, self.obsi_size_ls, ,self.opi_size_ls, self.hidden_nums = args
        self.hidden_nums = [128, 1]
        self.hyper_net1 = nn.Linear(self.input_size, self.num_agent_net * self.hidden_nums[0])
        self.hyper_net2 = nn.Linear(self.input_size, self.hidden_nums[0] * self.hidden_nums[1])
        self.agent_model = nn.ModuleList([Agent_net((self.obsi_size_ls[i], self.hidden_nums, self.opi_size_ls[i])) for i in range(self.num_agent_net)])
        self.optimizer = optim.Adam(self.parameters(), lr = 1e-3)

    def forward(self, inputs, inputs_ls):
        
        weights1 = torch.abs(self.hyper_net1(torch.FloatTensor(inputs)).view(-1, self.num_agent_net, self.hidden_nums[0]))
        weights2 = torch.abs(self.hyper_net2(torch.FloatTensor(inputs)).view(-1, self.hidden_nums[0], self.hidden_nums[1]))
        q_vals = q_vals.view(-1, 1, self.num_agent_net)
        q_tot =　torch.bmm(torch.bmm(q_vals, weights1), weights2).mean()
        return q_tot


