import torch
import numpy as np
import model
from torch import optim
from torch.nn import functional as F

GAMMA = 0.99
LEARNING_RATE = 1e-3
NOISY_SIGMA = 1


def calc_v_loss(batch, tgt_v_net1, tgt_v_net2, tgt_s_net, v_net1, v_net2, device='cpu'):
    states, actions, rewards, next_states, done = batch

    state_v = torch.FloatTensor(states).to(device)
    next_state_v = torch.FloatTensor(next_states).to(device)
    reward_v = torch.FloatTensor(rewards).to(device)
    action_v = torch.FloatTensor(actions).to(device)

    next_action = tgt_s_net(next_state_v)  # add some noisy
    noisy = np.random.normal(0, NOISY_SIGMA**2, next_action.shape[0])
    next_action = next_action + torch.FloatTensor(np.clip(noisy, -3*NOISY_SIGMA, 3*NOISY_SIGMA))

    value1, value2 = tgt_v_net1(next_state_v, next_action), tgt_v_net2(next_state_v, next_action)
    value = (0 if done else torch.minimum(value1, value2))

    expected_value = reward_v + GAMMA * value

    critic_value1, critic_value2 = v_net1(state_v, action_v), v_net2(state_v, action_v)

    return F.mse_loss(critic_value1, expected_value.detach()), F.mse_loss(critic_value2, expected_value.detach())


def calc_s_loss(batch, s_net, v_net, device='cpu'):
    states, actions, rewards, next_states, done = batch

    state_v = torch.FloatTensor(states).to(device)
    cur_actions_v = s_net(state_v)
    return -v_net(state_v, cur_actions_v).mean()


def update_tgt(target, source, x):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - x) * target_param.data + x * source_param.data)


if __name__ == '__main__':

    # env = Env()
    # buffer = ExperienceBuffer(REPLAY_SIZE)
    # agent = Agent(env, buffer)

    v_net1, v_net2, tgt_v_net1, tgt_v_net2 = model.Critic(2, 2), model.Critic(2, 2), model.Critic(2, 2), model.Critic(2,
                                                                                                                      2)
    s_net, tgt_s_net = model.Actor(2, 2), model.Actor(2, 2)

    act_opt = optim.Adam(s_net.parameters(), lr=LEARNING_RATE)
    crt1_opt, crt2_opt = optim.Adam(v_net1.parameters(), lr=LEARNING_RATE), optim.Adam(v_net2.parameters(),
                                                                                       lr=LEARNING_RATE)
    #while True:
