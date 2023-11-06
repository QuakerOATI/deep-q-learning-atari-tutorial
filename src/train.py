import numpy as np
from keras import optimizers

from .env import DISCOUNT, LEARNING_RATE, CLIPNORM

optimizer = optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIPNORM)


def train(env, optimizer):
    while True:
        state = np.array(env.reset())
