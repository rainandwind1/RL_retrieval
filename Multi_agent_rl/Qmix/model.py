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
        self.fc1 = nn.Linear(self.input_size, 64)
        self.gru_net = nn.GRU(64, self.hidden_size, batch_first = True)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, hidden_state, batch = False):
        fc1_op = self.fc1(inputs)
        fc1_op = fc1_op.unsqueeze(0)
        gru_op, hidden_s = self.gru_net(fc1_op, hidden_state)
        gru_op = gru_op.squeeze(0)
        q_val = self.fc2(gru_op)
        return q_val, hidden_s


    def get_action(self, inputs, hidden_state, epsilon, action_mask):
        inputs = inputs.unsqueeze(0)
        q_val, h_s = self(inputs, hidden_state)
        q_val = q_val.squeeze(0) * torch.FloatTensor(action_mask).to(hidden_state.device)
        for i in range(len(action_mask)):
            if action_mask[i] == 0:
                q_val[i] = -float('inf')
        seed = np.random.rand()
        if seed > epsilon:
            return torch.argmax(q_val, -1).item(), h_s
        else:
            avail_actions_ind = np.nonzero(action_mask)[0]
            action = np.random.choice(avail_actions_ind)
            return action, h_s


# 记忆库
class Replaybuffer():
    def __init__(self, args):
        self.size = args
        self.mem_list = collections.deque(maxlen=self.size)

    @ property
    def mem_size(self):
        return len(self.mem_list)

    def save_trans(self, trans):
        self.mem_list.append(trans)

    def sample_batch(self,  batch_size):
        episode_batch = random.sample(self.mem_list, batch_size)
        s_ep, a_ep, r_ep, s_next_ep, done_ep, obs_ep, obs_next_ep, a_pre_ep, action_mask_ep = ([] for i in range(9))
        for episode in episode_batch:
            s_ls, a_ls, r_ls, s_next_ls, done_ls, obs_ls, obs_next_ls, a_pre_ls, action_mask_ls = ([] for i in range(9))
            for trans in episode:
                s, a, r, s_next, done, obs, obs_next, a_pre, action_mask = trans
                s_ls.append(s)
                a_ls.append(a)
                r_ls.append([r])
                s_next_ls.append(s_next)
                done_ls.append([done])
                obs_ls.append(obs)
                obs_next_ls.append(obs_next)
                a_pre_ls.append(a_pre)
                action_mask_ls.append(action_mask)
            s_ep.append(s_ls)
            a_ep.append(a_ls)
            r_ep.append(r_ls)
            s_next_ep.append(s_next_ls)
            done_ep.append(done_ls)
            obs_ep.append(obs_ls)
            obs_next_ep.append(obs_next_ls)
            a_pre_ep.append(a_pre_ls)
            action_mask_ep.append(action_mask_ls)

        return s_ep, a_ep, r_ep, s_next_ep, done_ep, obs_ep, obs_next_ep, a_pre_ep, action_mask_ep



# Mix网络
class Mixing_net(nn.Module):  # s 为输入生成正权值
    def __init__(self, args):
        super(Mixing_net, self).__init__()
        self.num_agent_net, self.input_size, self.obsi_size_ls, self.opi_size_ls, self.device, self.lr = args
        self.hidden_nums = [64, 1]
        self.hyper_net1 = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_agent_net * self.hidden_nums[0])
        )
        self.hyper_net2 = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_nums[0] * self.hidden_nums[1])
        )
        self.hyper_net_b1 =  nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_nums[0])
        )
        self.hyper_net_b2 = nn.Sequential(
            nn.Linear(self.input_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_nums[1])
        )
        self.agent_model = nn.ModuleList([Agent_net(args = (self.obsi_size_ls[i] + self.opi_size_ls[i], 32, self.opi_size_ls[i])) for i in range(1)])
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

    def forward(self, q_vals, inputs):
        weights1 = torch.abs(self.hyper_net1(inputs).view(-1, self.num_agent_net, self.hidden_nums[0]))
        weights2 = torch.abs(self.hyper_net2(inputs).view(-1, self.hidden_nums[0], self.hidden_nums[1]))
        b1 = self.hyper_net_b1(inputs).view(-1, 1, self.hidden_nums[0])
        b2 = self.hyper_net_b2(inputs).view(-1, 1, self.hidden_nums[1])
        q_vals = q_vals.view(-1, 1, self.num_agent_net)
        q_tot = torch.bmm(torch.bmm(q_vals, weights1) + b1, weights2) + b2
        return q_tot


def trans_aidx(a_ls, action_size, device):
    a_vec = torch.zeros(len(a_ls), len(a_ls[0]), action_size).to(device)
    for idx, a_tot in enumerate(a_ls):
        for a_n in range(len(a_tot)):
            a_vec[idx][a_n][int(a_tot[a_n])] = 1.
    return a_vec

def trans_to_tensor(s_ls, a_ls, r_ls, s_next_ls, done_ls, obs_ls, obs_next_ls, a_pre_ls, action_mask_ls, device):
    return torch.FloatTensor(s_ls).to(device),\
            torch.tensor(a_ls, dtype = torch.int64).to(device),\
            torch.FloatTensor(r_ls).to(device),\
            torch.FloatTensor(s_next_ls).to(device),\
            torch.FloatTensor(done_ls).to(device),\
            torch.FloatTensor(obs_ls).to(device),\
            torch.FloatTensor(obs_next_ls).to(device),\
            torch.tensor(a_pre_ls, dtype = torch.int64).to(device),\
            np.array(action_mask_ls)



# def valid_filter(q_vals, action_mask):
#     q_vals = q_vals + mask
#     return q_vals



def train(replay_buffer, model, target_model, gamma, lr, batch_size):
    s_ep, a_ep, r_ep, s_next_ep, done_ep, obs_ep, obs_next_ep, a_pre_ep, action_mask_ep = replay_buffer.sample_batch(batch_size)
    # batch_size 个episode数据
    loss_tot = 0
    for s_ls, a_ls, r_ls, s_next_ls, done_ls, obs_ls, obs_next_ls, a_pre_ls, action_mask_ls in zip(s_ep, a_ep, r_ep, s_next_ep, done_ep, obs_ep, obs_next_ep, a_pre_ep, action_mask_ep):
        # print("episode training!  flag test:....")
        # calculate q_val && q_target
        a_vec = trans_aidx(a_ls, model.opi_size_ls[0], model.device)
        a_pre_vec = trans_aidx(a_pre_ls, model.opi_size_ls[0], model.device)
        s_ls, a_ls, r_ls, s_next_ls, done_ls, obs_ls, obs_next_ls, a_pre_ls, action_mask_ls = trans_to_tensor(s_ls, a_ls, r_ls, s_next_ls, done_ls, obs_ls, obs_next_ls, a_pre_ls, action_mask_ls, model.device)
        hidden_state = torch.zeros(1, 1, 32).to(model.device)
        qval_ls = []
        q_target_ls = []
        for i in range(model.num_agent_net):
            partical_inputs = torch.cat([obs_ls[:,i,:], a_pre_vec[:,i,:]], -1)
            q_idx, _ = model.agent_model[0](partical_inputs, hidden_state)
            q_val = torch.gather(q_idx, 1, a_ls[:,i].unsqueeze(-1))
            qval_ls.append(q_val)
            partical_next_inputs = torch.cat([obs_next_ls[:,i,:], a_vec[:,i,:]], -1)
            q_target, _ = target_model.agent_model[0](partical_next_inputs,  hidden_state)
            q_target[torch.FloatTensor(action_mask_ls[:,i,:]).to(model.device) == 0] = -9999999
            # q_target = valid_filter(q_target, action_mask_ls[:,i,:])   # action mask 过滤下一个动作
            q_target = r_ls + gamma * (torch.max(q_target, -1)[0]).unsqueeze(-1) * (1 - done_ls)
            q_target_ls.append(q_target)
        qval_ls = torch.cat(qval_ls, -1)
        q_target_ls = torch.cat(q_target_ls, -1)
        q_tot = model(qval_ls, s_ls) 
        q_target_tot = target_model(q_target_ls, s_next_ls).detach()
        loss_tot = ((q_target_tot - q_tot)**2).mean()
    loss_tot = loss_tot / batch_size
    model.optimizer.zero_grad()
    loss_tot.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1) 
    model.optimizer.step()
    
# def train(replay_buffer, model, target_model, gamma, lr, batch_size):
#     for i in range(4):
#         s_ep, a_ep, r_ep, s_next_ep, done_ep, obs_ep, obs_next_ep, a_pre_ep, action_mask_ep = replay_buffer.sample_batch(batch_size)
#         # batch_size 个episode数据
#         loss_tot = 0
#         for s_ls, a_ls, r_ls, s_next_ls, done_ls, obs_ls, obs_next_ls, a_pre_ls, action_mask_ls in zip(s_ep, a_ep, r_ep, s_next_ep, done_ep, obs_ep, obs_next_ep, a_pre_ep, action_mask_ep):
#             # print("episode training!  flag test:....")
#             # calculate q_val && q_target
#             s_ls, a_ls, r_ls, s_next_ls, done_ls, obs_ls, obs_next_ls, a_pre_ls, action_mask_ls = trans_to_tensor(s_ls, a_ls, r_ls, s_next_ls, done_ls, obs_ls, obs_next_ls, a_pre_ls, action_mask_ls)
#             a_vec = trans_aidx(a_ls, model.opi_size_ls[0])
#             a_pre_vec = trans_aidx(a_pre_ls, model.opi_size_ls[0])
#             hidden_state = torch.zeros(1, 1, 32)
#             qval_ls = []
#             q_target_ls = []
#             for i in range(model.num_agent_net):
#                 partical_inputs = torch.cat([obs_ls[:,i,:], a_pre_vec[:,i,:]], -1)
#                 q_idx, _ = model.agent_model[0](partical_inputs, hidden_state)
#                 q_val = torch.gather(q_idx, 1, torch.LongTensor(a_ls[:,i].unsqueeze(-1)))
#                 qval_ls.append(q_val)
#                 partical_next_inputs = torch.cat([obs_next_ls[:,i,:], a_vec[:,i,:]], -1)
#                 q_target, _ = target_model.agent_model[0](partical_next_inputs, hidden_state)
#                 q_target = valid_filter(q_target, action_mask_ls[:,i,:])   # action mask 过滤下一个动作
#                 q_target = r_ls + gamma * (torch.max(q_target, -1)[0]).unsqueeze(-1) * done_ls
#                 q_target_ls.append(q_target)
#             qval_ls = torch.cat(qval_ls, -1)
#             q_target_ls = torch.cat(q_target_ls, -1)
#             q_tot = model(qval_ls, s_ls) 
#             q_target_tot = target_model(q_target_ls, s_next_ls).detach()
#             loss_tot += ((q_target_tot - q_tot)**2).mean()
#         model.optimizer.zero_grad()
#         loss_tot.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) 
#         model.optimizer.step()
    
        

# class Colla_Q(nn.Module):
#     def __init__(self, args):
#         super(Colla_Q, self).__init__()



