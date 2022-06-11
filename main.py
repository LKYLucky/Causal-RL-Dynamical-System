import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from env import LotkaVolterraEnv
import torch


# import torch.nn.functional as F
# from stable_baselines.sac.policies import MlpPolicy
# from stable_baselines import SAC


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, env.action_space.n),
            torch.nn.ReLU(),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        probs = self.forward(state)

        # print('action probs:')
        # print(probs)

        # Equivalent to multinomial
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()
        return action


class Actor(torch.nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, env.action_space.n),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float)
        probs = self.forward(state)
        # Equivalent to multinomial
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()

        return action


class Critic(torch.nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )

    def forward(self, state):
        return self.net(state)

def optimal_policy(state):
    eps = 0.001

    if state[0]>state[1]+0.01:
        action = 1
    elif state[0]<state[1]-0.01:
        action = 2
    else:
        action = 0
    return action

def convert_to_vec(action):
    if action == 0:
        u = np.array([0, 0])
    elif action == 1:
        u = np.array([0, 1])
    elif action == 2:
        u = np.array([1, 0])

    return u



def discounted_rewards(rewards, gamma):
    returns = np.zeros_like(rewards)
    J = 0.0
    for t in reversed(range(len(rewards))):
        J = J * gamma + rewards[t]
        returns[t] = J
    return returns


def update_policy_reinforce(states, actions, returns, model, optimizer):
    loss = torch.zeros(1)

    for s in range(len(states)):
        state, action, J = states[s], actions[s], returns[s]
        print('state', state, ', action', torch.IntTensor.item(action), ', J', J)
        probs = model(state)  # https://pytorch.org/docs/stable/distributions.html
        m = torch.distributions.Categorical(probs=probs)
        log_prob = m.log_prob(action)
        loss -= log_prob * J

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def update_policy_a2c(states, actions, returns, actor, critic, actor_optimizer, critic_optimizer):
    actor_loss, critic_loss = torch.zeros(1), torch.zeros(1)

    for s in range(len(states)):
        state, action, R = states[s], actions[s], returns[s]
        probs = actor(state)  # https://pytorch.org/docs/stable/distributions.html
        policy = torch.distributions.Categorical(probs=probs)
        value = torch.IntTensor.item(critic(state))
        advantage = R - value

        actor_loss -= (policy.log_prob(action) * advantage)
        # loss for critic (MSE)
        critic_loss += torch.tensor(advantage, requires_grad=True).pow(2)

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()


def run(env, algorithm):
    model = Model()
    actor = Actor()
    critic = Critic()

    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr)

    print("obs space", env.observation_space)
    print("action_space", env.action_space.n)

    # train
    max_episode = 1000
    n_episode = 0
    max_step = 10
    scores = []
    prob_list = []

    while n_episode < max_episode:

        print('starting training episode %d' % n_episode)

        Z_init = np.array([1.0, 1.5])

        # Z_init = np.array([1.0, 1.5]) # np.ones((N,))
        env.init_state = Z_init
        Z_history = np.expand_dims(Z_init, 0)

        # done = False
        states = []
        actions = []
        rewards = []
        observation = torch.tensor(env.reset(), dtype=torch.float)

        for i in range(max_step):
            if algorithm == "reinforce":
                action = model.select_action(observation)
            elif algorithm == "a2c":
                action = actor.select_action(observation)

            elif algorithm == "optimal policy":
                action = optimal_policy(observation)


            # convert action to vectors (0,0),(0,1),(1,0)
            u = convert_to_vec(action)
            obs, reward, _, _, Z = env.step(u)
            print("state", observation, ", action", action, ", reward", reward)
            Z_history = np.concatenate((Z_history, Z), 0)
            states.append(observation)

            actions.append(torch.tensor(action, dtype=torch.int))

            rewards.append(reward)
            observation = torch.tensor(obs, dtype=torch.float)

        x = torch.tensor([1, 2], dtype=torch.float)
        y = model.forward(x)
        y = y.tolist()
        prob_list.append(y)
        scores.append(sum(rewards))
        # prob_list.append([zero_prob, one_prob, two_prob])
        n_episode += 1

        returns = discounted_rewards(rewards, gamma=0.9)
        if algorithm == "reinforce":
            update_policy_reinforce(states, actions, returns, model, optimizer)
        elif algorithm == "a2c":
            update_policy_a2c(states, actions, returns, actor, critic, actor_optimizer, critic_optimizer)

    # eval -- let's make this a separate function, analogous to 'run' but without any training or policy updating
    # done = False

    #eval -- let's make this a separate function, analogous to 'run' but without any training or policy updating
    #done = False
    observation = torch.tensor(env.reset(), dtype=torch.float)
    #env.render()

    rewards = []
    total_rewards = 0
    Z_init = np.array([1.0, 1.5]) #np.random.uniform(0.5,2.0,size=(2)) # randomly sampled np.array([1.0, 1.5]) # np.ones((N,))
    env.init_state = Z_init
    Z_history = np.expand_dims(Z_init, 0)
    for i in range(max_step):#while not done:

        if algorithm == "reinforce":
            action = model.select_action(observation)
        elif algorithm == "a2c":
            action = actor.select_action(observation)

        u = convert_to_vec(action)
        obs, reward, _, _, Z = env.step(u)
        Z_history = np.concatenate((Z_history, Z), 0)
        observation = torch.tensor(obs, dtype=torch.float)
        rewards.append(reward)
        total_rewards += reward
        #env.render()

    print('reward', total_rewards)
    print("p_list",prob_list)

    plt.figure()
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('./rewards.png')
    #plt.show()

    tt = np.linspace(0, max_step, Z_history.shape[0])
    plt.cla()
    plt.plot(tt, Z_history[:, 0], 'b', label='prey')
    plt.plot(tt, Z_history[:, 1], 'r', label='predators')
    plt.plot(tt, Z_history[:, 0]/Z_history[:, 1], 'k', label='ratio')
    plt.legend(shadow=True, loc='upper left')
    plt.xlabel('t')
    plt.ylabel('n')
    plt.savefig('./state_trajectory.png')
    #plt.show()


    tt = np.linspace(0, max_episode, max_episode)
    plt.cla()
    zeros_prob = [item[0] for item in prob_list]
    ones_prob = [item[1] for item in prob_list]
    twos_prob = [item[2] for item in prob_list]
    plt.plot(tt, zeros_prob, 'kx', label='Item 0')
    plt.plot(tt, ones_prob, 'rx', label='Item 1')
    plt.plot(tt, twos_prob, 'o', label='Item 2')
    plt.legend(shadow=True, loc='lower right')
    plt.xlabel('Episode #')
    plt.ylabel('Probabilities')
    plt.savefig('./policy_probs.png')
    plt.show()

    exit()
    env.close()

N = 2 # number of species
tau = 1
dt = 1e-2
env = LotkaVolterraEnv(N, tau, dt)

#run(env, algorithm = "reinforce")
#run(env, algorithm = "a2c")
run(env, algorithm = "optimal policy")