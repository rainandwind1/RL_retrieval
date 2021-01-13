import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import collections
import random


# Qmix 的 agent 网络
class QMIXAgent_net(nn.Module):
    def __init__(self, args):
        super(QMIXAgent_net, self).__init__()
        self.input_size, self.hidden_size, self.action_size, self.device = args
        self.fc1 = nn.Linear(self.input_size, 64)
        self.gru_net = nn.GRU(64, self.hidden_size, batch_first = True)
        self.fc2 = nn.Linear(self.hidden_size, self.action_size)

    def forward(self, inputs, hidden_s, max_step = 1):
        fc1_op = self.fc1(inputs)
        fc1_op = fc1_op.view(-1, max_step, 64)
        gru_op, hidden_next = self.gru_net(fc1_op, hidden_s)
        gru_op = gru_op.view(-1, max_step, self.hidden_size)
        q_val = self.fc2(gru_op).view(-1, max_step, self.action_size)
        return q_val, hidden_next

    def get_action(self, obs, hidden_s, epsilon, action_mask):
        inputs = obs.unsqueeze(0)
        q_val, h_s = self(inputs, hidden_s)
        q_val = q_val.squeeze(0)
        q_val = q_val.squeeze(0) * torch.FloatTensor(action_mask).to(self.device)
        for i in range(len(action_mask)):
            if action_mask[i] == 0:
                q_val[i] = -float('inf')
        seed = np.random.rand()
        if seed > epsilon:
            return torch.argmax(q_val).item(), h_s
        else:
            avail_actions_ind = np.nonzero(action_mask)[0]
            action = np.random.choice(avail_actions_ind)
            return action, h_s

# Qmix 的 mix 网络
class QMIXMixing_net(nn.Module):
    def __init__(self, args):
        super(QMIXMixing_net, self).__init__()
        self.num_agent, self.joint_obs_size, self.obs_info, self.action_info, self.device, self.lr = args
        self.hidden_nums = [64, 1]
        # 超网络 用于生成混合加权各个代理的q值的权重
        self.hyper_netw1 = nn.Linear(self.joint_obs_size, self.num_agent * self.hidden_nums[0])
        self.hyper_netw2 = nn.Linear(self.joint_obs_size, self.hidden_nums[0] * self.hidden_nums[1])
        self.hyper_netb1 = nn.Linear(self.joint_obs_size, self.hidden_nums[0])
        self.hyper_netb2 = nn.Linear(self.joint_obs_size, self.hidden_nums[1])
        self.agent_model = nn.ModuleList([QMIXAgent_net(args = (self.obs_info[i] + self.action_info[i], 32, self.action_info[i], self.device)) for i in range(self.num_agent)])
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def forward(self, q_vals, inputs):
        weights1 = torch.abs(self.hyper_netw1(inputs)).view(-1, self.num_agent, self.hidden_nums[0])
        weights2 = torch.abs(self.hyper_netw2(inputs)).view(-1, self.hidden_nums[0], self.hidden_nums[1])
        b1 = self.hyper_netb1(inputs).view(-1, 1, self.hidden_nums[0])
        b2 = self.hyper_netb2(inputs).view(-1, 1, self.hidden_nums[1])

        q_vals = q_vals.view(-1, 1, self.num_agent)
        q_tot = torch.bmm(torch.bmm(q_vals, weights1) +  b1, weights2) + b2
        return q_tot


# 记忆库
class Replaybuffer():
    def __init__(self, args):
        self.size = args
        self.mem_list = collections.deque(maxlen = self.size)

    @property
    def mem_len(self):
        return len(self.mem_list)
    
    def save_trans(self, trans):
        self.mem_list.append(trans)
    
    def sample_batch(self, batch_size = 64):
        episode_batch = random.sample(self.mem_list, batch_size)
        s_ep, a_ep, a_onehot_ep, r_ep, s_next_ep, done_ep, obs_ep, obs_next_ep, a_pre_ep, a_pre_onehot_ep, action_mask_ep, loss_mask_ep = ([] for _ in range(12))
        for episode in episode_batch:
            s_ls, a_ls, a_onehot_ls, r_ls, s_next_ls, done_ls, obs_ls, obs_next_ls, a_pre_ls, a_pre_onehot_ls, action_mask_ls, loss_mask_ls = ([] for _ in range(12))
            for trans in episode:
                s, a, a_onehot, r, s_next, done, obs, obs_next, a_pre, a_pre_onehot, action_mask, loss_mask = trans
                s_ls.append(s)
                a_ls.append(a)
                a_onehot_ls.append(a_onehot)
                r_ls.append([r])
                s_next_ls.append(s_next)
                done_ls.append([done])
                obs_ls.append(obs)
                obs_next_ls.append(obs_next)
                a_pre_ls.append(a_pre)
                a_pre_onehot_ls.append(a_pre_onehot)
                action_mask_ls.append(action_mask)
                loss_mask_ls.append([loss_mask])
            s_ep.append(s_ls)
            a_ep.append(a_ls)
            a_onehot_ep.append(a_onehot_ls)
            r_ep.append(r_ls)
            s_next_ep.append(s_next_ls)
            done_ep.append(done_ls)
            obs_ep.append(obs_ls)
            obs_next_ep.append(obs_next_ls)
            a_pre_ep.append(a_pre_ls)
            a_pre_onehot_ep.append(a_pre_onehot_ls)
            action_mask_ep.append(action_mask_ls)
            loss_mask_ep.append(loss_mask_ls)

        return s_ep, a_ep, a_onehot_ep, r_ep, s_next_ep, done_ep, obs_ep, obs_next_ep, a_pre_ep, a_pre_onehot_ep, action_mask_ep, loss_mask_ep


