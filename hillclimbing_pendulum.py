""" 
Member in group
6288024 Komsan      Kongwongsupak
6288060 Anyamanee   Amatyakul
6288071 Kasidis     Chokpaiboon
6288079 Cholravee   Kittimethee

Project topic: Solving OpenAI's Gym Classic environments using Hill-climbing search (sideway + restart)
- One member per environment
- CartPole-v1
- Arcobat-v1
- MountainCar-v0
- Pendulum-v0
"""


"""A module for local search algorithm to find a good pendulum agent."""
import sys

import numpy as np
import gym


# def sigmoid(x):
#     """Return sigmoid value of x."""
#     return 1. / (1. + np.exp(-x))


class PDAgent:
    """pendulum agent."""

    def __init__(self, std=0.1, w1=None, b1=None):
        """Create a pendulum agent with a randomly initialized weight."""
        self.std = std
        self.w1 = w1
        self.b1 = b1
        if self.w1 is None:
            self.w1 = np.random.normal(0, self.std, [3])    # cos(), sin(), v
        if self.b1 is None:
            self.b1 = np.random.normal(0, self.std, [1])

    def act(self, obs):
        """Return an action from the observation."""
        # move = sigmoid(np.dot(obs, self.w1) + self.b1)
        move = np.dot(obs, self.w1) + self.b1
        
        # if move > 2:
        #     move = 2
        # elif move < -2:
        #     move = -2
        # print(move)
        return move

    def neighbors(self, step_size=0.01):
        """Return all neighbors of the current agent."""
        neighbors = []
        for i in range(3):
            w1 = self.w1.copy()
            w1[i] += step_size
            neighbors.append(PDAgent(
                self.std, w1=w1, b1=self.b1))
            w1 = self.w1.copy()
            w1[i] -= step_size
            neighbors.append(PDAgent(
                self.std, w1=w1, b1=self.b1))
        b1 = self.b1.copy()
        b1 += step_size
        neighbors.append(PDAgent(self.std, w1=self.w1, b1=b1))
        b1 = self.b1.copy()
        b1 -= step_size
        neighbors.append(PDAgent(self.std, w1=self.w1, b1=b1))
        return neighbors


    def __repr__(self):
        """Return a weights of the agent."""
        out = [f'{w:.3}' for w in self.w1] + [f'{self.b1[0]:.3}']
        return ', '.join(out)

    def __eq__(self, agent):
        """Return True if agents has the same weights."""
        if isinstance(agent, PDAgent):
            return np.all(self.b1 == agent.b1) and np.all(self.w1 == agent.w1)
        return False

    def __hash__(self):
        """Return hash value."""
        return hash((*self.w1, *self.b1))


def simulate(env, agents, repeat=1, max_iters=200):
    """Simulate pendulum for all agents, and return rewards of all agents."""
    rewards = [0 for __ in range(len(agents))]
    for i, agent in enumerate(agents):
        total_reward = 0
        for __ in range(repeat):
            env.seed(42)
            obs = env.reset()
            for t in range(max_iters):
                action = agent.act(obs)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                if done:
                    break
        rewards[i] = total_reward / repeat
    return np.array(rewards)


def hillclimb(env, agent, max_iters=10000):
    """Run a hill-climbing search, and return the final agent."""
    cur_agent = agent
    cur_r = simulate(env, [agent])[0]

    history = [cur_r]
    explored = set()
    explored.add(cur_agent)
    for __ in range(max_iters):
        env.render()
        neighbors = cur_agent.neighbors()
        _n = []
        for a in neighbors:
            if a not in explored:  # we do not want to move to previously explored ones.
                _n.append(a)
        neighbors = _n
        rewards = simulate(env, neighbors)
        best_i = np.argmax(rewards)
        history.append(rewards[best_i])
        if rewards[best_i] <= cur_r:
            return cur_agent, history
        cur_agent = neighbors[best_i]
        cur_r = rewards[best_i]
    return cur_agent, history


def hillclimb_sideway(env, agent, max_iters=10000, sideway_limit=10):
    """
    Run a hill-climbing search, and return the final agent.

    Parameters
    ----------
    env : OpenAI Gym Environment.
        A cart-pole environment for the agent.
    agent : CPAgent
        An initial agent.
    max_iters: int
        Maximum number of iterations to search.
    sideway_limit
        Number of sideway move to make before terminating.
        Note that the sideway count reset after a new better neighbor
        has been found.

    Returns
    ----------
    final_agent : CPAgent
        The final agent.
    history : List[float]
        A list containing the scores of the best neighbors of
        all iterations. It must include the last one that causes
        the algorithm to stop.

    """
    cur_agent = agent
    cur_r = simulate(env, [agent])[0]

    explored = set()
    explored.add(cur_agent)
    history = [cur_r]

    for __ in range(max_iters):
        env.render()
        # TODO 1: Implement hill climbing search with sideway move.
        neighbors = cur_agent.neighbors()
        _n = []
        for a in neighbors:
            if a not in explored:  # we do not want to move to previously explored ones.
                _n.append(a)
        neighbors = _n
        rewards = simulate(env, neighbors)
        best_i = np.argmax(rewards)
        history.append(rewards[best_i])
        if rewards[best_i] < cur_r:
            return cur_agent, history
        #   Sideway move
        elif rewards[best_i] == cur_r:  
            for __ in range(sideway_limit):
                rewards = simulate(env, neighbors)
                equal_i = np.argmax(rewards)
                history.append(rewards[equal_i])
                if rewards[equal_i] < cur_r:
                    best_i = equal_i
                    break
            return cur_agent, history
        # 
        cur_agent = neighbors[best_i]
        cur_r = rewards[best_i]

        # pass
    return cur_agent, history


def hillclimb_restart(env, agent):
    """Run a hill-climbing search, and return the final agent."""
    best_agent, rewards = hillclimb_sideway(env, agent)
    best_reward = max(rewards)
    for __ in range(30):
        cur_agent = PDAgent()
        env.reset()
        temp_agent, history = hillclimb_sideway(env, cur_agent)
        reward = max(history)
        if best_reward < reward:
            best_agent = temp_agent
            best_reward = reward
                
    return best_agent, history  


def simulated_annealing(env, agent, init_temp=25.0, temp_step=-0.1, max_iters=10000):
    """
    Run a hill-climbing search, and return the final agent.

    Parameters
    ----------
    env : OpenAI Gym Environment.
        A pendulum environment for the agent.
    agent : PDAgent
        An initial agent.
    init_temp : float
        An initial temperature.
    temp_step : float
        A step size to change the temperature for each iteration.
    max_iters: int
        Maximum number of iterations to search.

    Returns
    ----------
    final_agent : PDAgent
        The final agent.
    history : List[float]
        A list containing the scores of the sampled neighbor of
        all iterations.

    """
    cur_agent = agent
    cur_r = simulate(env, [agent])[0]
    history = [cur_r]
    sideway = 0

    for __ in range(max_iters):
        # TODO 2: Implement simulated annealing search.
        # We should not keep track of "already explored" neighbor.
        pass


    return cur_agent, history


if __name__ == "__main__":
    gym.envs.register(
        id="Pendulum-v1",
        entry_point="gym.envs.classic_control:PendulumEnv",
        max_episode_steps=200,
    )
    env = gym.make('Pendulum-v1')
    # w1 = np.array([-0.0723, -0.0668, 0.151, 0.0802])
    # b1 = np.array([-0.0214])
    if len(sys.argv) > 1:
        if sys.argv[1] != 'random':
            _w = [float(v.strip()) for v in sys.argv[1].split(',')]
            w1 = np.array(_w[:3])
            b1 = np.array(_w[3:4])
            agent = PDAgent(w1=w1, b1=b1)
        else:
            agent = PDAgent()
        print(agent)
        env.seed(42)
        obs = env.reset()
        total_reward = 0
        for t in range(1500):
            env.render()
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            # if done:
            #     break

        print('Total Reward: ', total_reward)
    else:
        agent = PDAgent()
        # Hill Climbing search can solve this case.
        # agent = PDAgent(w1=np.array([0.0111, 0.0909, 0.0688, 0.189]), b1=np.array([0.0456]))
        # Hill Climbing search cannot solve this case, but sideway move limit at 10 will solve this.
        # agent = PDAgent(w1=np.array([0.0155, 0.0946, 0.0225, 0.0975]), b1=np.array([-0.0628]))
        initial_reward = simulate(env, [agent])[0]
        print('Initial:    ', agent, ' --> ', f'{initial_reward:.5}')
        # agent, history = hillclimb(env, agent)
        # agent, history = hillclimb_sideway(env, agent)
        agent, history = hillclimb_restart(env, agent)
        initial_reward = simulate(env, [agent])[0]
        for score in history:
            print(score)
        print('After:      ', agent, ' --> ', f'{initial_reward:.5}')

        neighbors = agent.neighbors()
        rewards = simulate(env, neighbors)
        for i, (a, r) in enumerate(zip(neighbors, rewards)):
            print(f'Neighbor {i}: ', a, ' --> ', f'{r:.5}')
    env.close()
