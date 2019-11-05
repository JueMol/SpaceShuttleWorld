# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 19:09:30 2019

@author: Juergen Mollen (Git:JueMol)  (inspired by https://keon.io/deep-q-learning/)
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
import random
from collections import deque

#########################
# Deep Q-learning Agent #
#########################
class DQNAgent:
    def __init__(self, state_size, action_size, decay=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95   # discount rate
        self.epsilon = 1.   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = decay
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='tanh', bias_regularizer=l2(0.001), use_bias=True))
        #model.add(Dropout(0.0001))
        model.add(Dense(128, activation='tanh', bias_regularizer=l2(0.001), use_bias=True))
        #model.add(Dropout(0.0001))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        act_values = self.model.predict(state)

        if np.random.rand() <= self.epsilon:
            return(random.randrange(self.action_size))
        
        action = np.argmax(act_values[0])
        
        if action < 0:
            action += self.action_size

        if action >= self.action_size:
            action -= self.action_size
        
        return action
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
              
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay