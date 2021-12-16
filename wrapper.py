import collections
import sqlite3
import numpy as np
import torch

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class Env:
    def __init__(self):
        self.action_space = 10  # 50
        self.state = None
        self.reward = None
        self.action = None
        self.state_id = None
        self.done = False

    def state_comm(self):
        while True:
            connect = sqlite3.connect('../db.sqlite3')
            cursor = connect.cursor()
            cursor.execute("SELECT * FROM env_experience ORDER BY id DESC LIMIT 1")
            result = cursor.fetchone()

            if result is not None:
                if result[0] != self.state_id:
                    # if db update happened, update state and reward
                    self.state_id = result[0]
                    self.state = eval(result[1])
                    self.reward = float(result[2])
                    self.done = result[5]

            connect.close()

            if self.state is not None and self.reward is not None:
                return self.state, self.reward, self.done

    def action_comm(self, action):
        # store the action
        self.action = action
        connect = sqlite3.connect('../db.sqlite3')
        cursor = connect.cursor()

        act = "UPDATE env_experience set action = " + str(self.action) + " where ID=" + str(self.state_id)
        cursor.execute(act)
        connect.commit()

        connect.close()
        self.reset()

        return self.state_comm()

    def sample_action(self):
        return np.random.choice(range(self.action_space), 1)[0]

    def reset(self):
        self.state = self.reward = self.action = None


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self.state = None
        self.total_reward = None
        self.reset()

    def step(self, net, device="cpu"):
        done_reward = None

        state_a = np.array([self.state], copy=False)
        state_v = torch.tensor(state_a).to(device)
        state_v = state_v.float()
        action_v = net(state_v)

        action = float(action_v.item())

        new_state, reward, is_done = self.env.action_comm(action)
        #print(new_state)
        self.total_reward += reward
        #print(is_done)
        if is_done == "True":
            is_done = True
        else:
            is_done = False

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self.reset()

        return done_reward

    def reset(self):
        self.state, self.total_reward, _ = self.env.state_comm()


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)

# env.state_comm()
# env.action_comm(1919)

# while True:
#
#     print(114514)
#     time.sleep(0.1)
