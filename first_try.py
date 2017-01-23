import gym
import sklearn.neural_network as net
import random
from numpy import array, reshape
import matplotlib.pyplot as plt
import os, sys

def create_data_set(iter_num = 100):
    
    X = []
    Y = []
    
    env = gym.make('CartPole-v0')

    for _ in range(iter_num):

        life = 0
        is_dead = False
        env.reset()
        while not is_dead:
            step = random.randint(0, 1)
            curr_state = array(env.state)
            curr_state.resize(5)
            curr_state[4] = step
            X.append(curr_state)

            is_dead = env.step(step)[2]
            life += int(not is_dead)

        Y += range(life, -1, -1)

    return X, Y


    
def create_regressor(X, Y, factor = 30):
    rgs = net.MLPRegressor((30,30,30), activation = 'logistic')
    rgs.fit(X, Y)

    return rgs

def play_with_regressor(rgs, show = False):
    env = gym.make('CartPole-v0')
    env.reset()

    done = False
    points = 0
    
    while not done:
        curr_state = array(env.state)
        curr_state.resize(5)

        curr_state[4] = 0
        grd0 = rgs.predict(reshape(curr_state, (1, -1)))

        curr_state[4] = 1
        grd1 = rgs.predict(reshape(curr_state, (1, -1)))

        if show:
            env.render()
            
        if grd1 > grd0:
            done = env.step(1)[2]
        else:
            done = env.step(0)[2]

        points += int(not done)

    env.close()
    return points
