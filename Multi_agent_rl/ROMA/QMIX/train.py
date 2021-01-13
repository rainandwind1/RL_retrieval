import torch
from torch import nn, optim
import torch.nn.functional as F
from utils import *
from model import *

# 训练函数
def QMIXtrain(replay_buffer, model, target_model, gamma, lr = 1e-3, batch_size = 64):
    episode_batch = replay_buffer.sample_batch(batch_size)
    s_ep, a_ep, a_onehot_ep, r_ep, s_next_ep, done_ep, obs_ep, obs_next_ep, a_pre_ep, a_pre_onehot_ep, action_mask_ep, loss_mask_ep, max_steps = trans_to_tensor(episode_batch, model.device)
    hidden_s = torch.zeros(1, batch_size, model.agent_model[0].hidden_size).to(model.device)
    q_val_ls = []
    q_target_ls = []
    for i in range(model.num_agent):
        agent_inputs = torch.cat([obs_ep[:, :, i, :], a_pre_onehot_ep[:, :, i, :]], -1) # batch_size, seq_len, n_agent, obs_size // batch_size, seq_len, n_agent, action_size
        q_idx, _ = model.agent_model[i](agent_inputs, hidden_s, max_step = max_steps)
        q_val = torch.gather(q_idx, -1, a_ep[:, :, i].unsqueeze(-1))
        q_val_ls.append(q_val)
        
        agent_next_inputs = torch.cat([obs_next_ep[:, :, i, :], a_onehot_ep[:, :, i, :]], -1)
        q_target, _ = target_model.agent_model[i](agent_next_inputs, hidden_s, max_step = max_steps)
        q_target = valid_filter(q_target, action_mask_ep[:, :, i, :])
        q_target = r_ep + gamma * (torch.max(q_target, -1)[0]).unsqueeze(-1) * (1 - done_ep)
        q_target_ls.append(q_target)
    qval_ls = torch.cat(q_val_ls, -1)
    q_target_ls = torch.cat(q_target_ls, -1)
    q_tot = model(qval_ls, s_ep).view(batch_size, -1, 1) 
    q_target_tot = target_model(q_target_ls, s_next_ep).view(batch_size, -1, 1) .detach()
    loss_tot = (((q_target_tot - q_tot)**2) * loss_mask_ep).mean()
    
    # optimize
    model.optimizer.zero_grad()
    loss_tot.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1) 
    model.optimizer.step()

