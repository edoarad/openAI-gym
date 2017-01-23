import gym
import math
import random


class agent(object):

    def __init__(self, env = 'CartPole-v0', predictor = None):
        self.env = gym.make(env)
        self.parameters = None
        self.num_episode = 0
        
        if predictor == None:
            self.predictor = self.random_predictor
        else:
            self.predictor = predictor
        
        log = open("log.txt", "w")
        log.truncate()
        log.close()
    
    def choose_best_action(self, observation):
        maximal_action = None
        maximal_reward = -math.inf

        for action in range(self.env.action_space.n):
            expected_reward_after_action = self.predictor(observation, action)
            if expected_reward_after_action > maximal_reward:
                maximal_reward = expected_reward_after_action
                maximal_action = action
        
        return maximal_action
    
    '''
    Learn the parameters, using learn_iteration() which should be supplied to the agent.
    '''
    def learn():
        pass
    
    def learn_iteration():
        pass
    
    '''
    Run the environment with the current parameters.
    Return the total reward.
    '''
    def run_episode(self, env = None, parameters = None, display = False, max_steps = 1000):
        if env == None:
            env = self.env
        if parameters == None:
            parameters = self.parameters
        predictor = self.predictor
        self.num_episode += 1
        log = open("log.txt", "w")
        log.write("Episode {}\n".format(self.num_episode)) 
        
        observation = env.reset()
        total_reward = 0
        for step in range(max_steps):
            if display: env.render()
            action = self.choose_best_action(observation)
            observation, reward, done, info = env.step(action) 
            log.write("{:4}: observation, reward, done, info = {}, {}, {}, {}\n".format(step, observation, reward, done, info))
            total_reward += reward
            if (done):
                break
        
        log.write("\tTotal reward for episode is: {}\n".format(total_reward))
        log.close()
        return total_reward

    def random_predictor(self, observation, action):
        return random.random() # a random value

