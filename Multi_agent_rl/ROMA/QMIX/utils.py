

def trans_to_tensor(episode_batch, device):
    s_ep, a_ep, a_onehot_ep, r_ep, s_next_ep, done_ep, obs_ep, obs_next_ep, a_pre_ep, a_pre_onehot_ep, action_mask_ep, loss_mask_ep = episode_batch
    max_ep_len = 0
    for epi in s_ep:
        max_ep_len = max(max_ep_len, len(epi))
    for i in range(len(s_ep)):
        s_ep[i] += [[0.] * len(s_ep[0][0])] * (max_ep_len - len(s_ep[i]))
        a_ep[i] += [[0.] * len(a_ep[0][0])] * (max_ep_len - len(a_ep[i]))
        a_onehot_ep[i] += [[[0.] * len(a_onehot_ep[0][0][0]) for _ in range(len(a_onehot_ep[0][0]))]] * (max_ep_len - len(a_onehot_ep[i]))
        r_ep[i] += [[0.]] * (max_ep_len - len(r_ep[i]))
        s_next_ep[i] += [[0.] * len(s_next_ep[0][0])] * (max_ep_len - len(s_next_ep[i]))
        done_ep[i] += [[0.]] * (max_ep_len - len(done_ep[i]))
        obs_ep[i] += [[0.] * len(obs_ep[0][0][0]) for _ in range(len(obs_ep[0][0]))] * (max_ep_len - len(obs_ep[i]))
        obs_next_ep[i] += [[0.] * len(obs_next_ep[0][0][0]) for _ in range(len(obs_next_ep[0][0]))] * (max_ep_len - len(obs_next_ep[i]))
        a_pre_ep[i] += [[0.] * len(a_pre_ep[0][0])] * (max_ep_len - len(a_pre_ep[i]))
        a_pre_onehot_ep[i] += [[0.] * len(a_pre_onehot_ep[0][0][0]) for _ in range(len(a_pre_onehot_ep[0][0]))] * (max_ep_len - len(a_pre_onehot_ep[i]))
        action_mask_ep[i] += [[0.] * len(action_mask_ep[0][0][0]) for _ in range(len(action_mask_ep[0][0]))] * (max_ep_len - len(action_mask_ep[i]))
        loss_mask_ep[i] += [[0.]] * (max_ep_len - len(loss_mask_ep[i]))

    return torch.FloatTensor(s_ep).to(device),\
        torch.LongTensor(a_ep).to(device),\
        torch.FloatTensor(a_onehot_ep).to(device),\
        torch.FloatTensor(r_ep).to(device),\
        torch.FloatTensor(s_next_ep).to(device),\
        torch.FloatTensor(done_ep).to(device),\
        torch.FloatTensor(obs_ep).to(device),\
        torch.FloatTensor(obs_next_ep).to(device),\
        torch.LongTensor(a_pre_ep).to(device),\
        torch.FloatTensor(a_pre_onehot_ep).to(device),\
        torch.FloatTensor(action_mask_ep).to(device),\
        torch.FloatTensor(loss_mask_ep).to(device)


def valid_filter(q_vals, action_mask):
    mask = torch.ones_like(action_mask) * -999 * (1 - action_mask)
    q_vals = q_vals * mask
    return q_vals


