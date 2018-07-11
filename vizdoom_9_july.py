
# coding: utf-8

# In[1]:

#!/usr/bin/env python


from __future__ import division
from __future__ import print_function
import keras as keras
from vizdoom import *
import itertools as it
from random import sample, randint, random
from time import time, sleep
import numpy as np
from keras_sfa import *
import skimage.color, skimage.transform
import tensorflow as tf
from keras.callbacks import LearningRateScheduler, Callback
from keras_vizdoom_controller import *
from tqdm import trange

# Q-learning settings
learning_rate = 0.0001
# learning_rate = 0.0001
discount_factor = 0.95
epochs = 20
learning_steps_per_epoch = 2000
replay_memory_size = 10000
# Note: Health gathering settings
#screen_resolution = RES_84x84
#screen_format = GRAY8


# NN learning settings
batch_size = 64

class History(Callback):
    """
    Callback that records events into a `History` object.

    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.
    """

    def on_train_begin(self, logs=None):
        if not hasattr(self, 'epoch'):
            self.epoch = []
            self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


# SFA learning settings
W = np.zeros((batch_size, batch_size))

for i in range(batch_size):
    for j in range(batch_size):
        if abs(i-j) < 2 and i<j and i%2 is 0 and j%2 is 1:
            W[i, j] = 1
            W[j, i] = 1

# Training regime
test_episodes_per_epoch = 10

# Other parameters
frame_repeat = 6
resolution = (84, 84)
episodes_to_watch = 10

model_savefile = "/tmp/model.ckpt"
save_model = True
load_model = False
skip_learning = False
# Configuration file path
#config_file_path = "../../scenarios/simpler_basic.cfg"
#config_file_path = "../../scenarios/take_cover.cfg"
config_file_path = "../../scenarios/health_gathering.cfg"
#config_file_path = "../../scenarios/deadly_corridor.cfg"
#config_file_path = "../../scenarios/rocket_basic.cfg"
#config_file_path = "../../scenarios/basic.cfg"


# In[ ]:

# Converts and down-samples the input image
def preprocess(img):
    img = skimage.transform.resize(img, resolution)
    img = img.astype(np.float32)
    return img


class ReplayMemory:
    def __init__(self, capacity):
        channels = 1
        state_shape = (capacity, resolution[0], resolution[1], channels)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.isterminal = np.zeros(capacity, dtype=np.float32)

        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward):
        self.s1[self.pos, :, :, 0] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, 0] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_sample(self, sample_size):
        i = sample(range(0, self.size), sample_size)
        return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i]


def learn_from_memory():
    """ Learns from a single transition (making use of replay memory).
    s2 is ignored if s2_isterminal """

    # Get a random minibatch from the replay memory and learns from it.
    if memory.size > batch_size:
        s1, a, s2, isterminal, r = memory.get_sample(batch_size)

        q2 = np.max(keras_controller.predict(s2), axis=1)#(get_q_values(s2), axis=1)

        target_q = keras_controller.predict(s1)#(s1)
        # target differs from q only for the selected action. The following means:
        # target_Q(s,a) = r + gamma * max Q(s2,_) if isterminal else r
        target_q[np.arange(target_q.shape[0]), a] = r + discount_factor * (1 - isterminal) * q2
        keras_controller.fit(s1, target_q, epochs=1, batch_size=batch_size, verbose = 0, callbacks=[keras.callbacks.History()])
#History])


def perform_learning_step(epoch):
    """ Makes an action according to eps-greedy policy, observes the result
    (next state, reward) and learns from the transition"""

    def exploration_rate(epoch):
        """# Define exploration rate change over time"""
        start_eps = 1.0
        end_eps = 0.1
        const_eps_epochs = 0.1 * epochs  # 10% of learning time
        eps_decay_epochs = 0.6 * epochs  # 60% of learning time

        if epoch < const_eps_epochs:
            return start_eps
        elif epoch < eps_decay_epochs:
            # Linear decay
            return start_eps - (epoch - const_eps_epochs) /                                (eps_decay_epochs - const_eps_epochs) * (start_eps - end_eps)
        else:
            return end_eps

    s1 = preprocess(game.get_state().screen_buffer)

    # With probability eps make a random action.
    eps = exploration_rate(epoch)
    if random() <= eps:
        a = randint(0, len(actions) - 1)
    else:
        # Choose the best action according to the network.
#        a = get_best_action(s1)
        a = np.argmax(keras_controller.predict(np.reshape(s1, (1, s1.shape[0], s1.shape[1], 1))))
    reward = game.make_action(actions[a], frame_repeat)

    isterminal = game.is_episode_finished()
    s2 = preprocess(game.get_state().screen_buffer) if not isterminal else None

    # Remember the transition that was just experienced.
    memory.add_transition(s1, a, s2, isterminal, reward)

    learn_from_memory()


# Creates and initializes ViZDoom environment.
def initialize_vizdoom(config_file_path):
    print("Initializing doom...")
    game = DoomGame()
    game.load_config(config_file_path)
    game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.GRAY8)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.init()
    print("Doom initialized.")
    return game

def get_sfa_batch_from_memory(batch_size, memory):
    s1, a, s2, isterminal, r = memory.get_sample(batch_size*3)
    riffled = np.zeros((sfa_batch_size, s1.shape[1], s1.shape[2], s1.shape[3]))
    not_terminal_indices =  [i for i in range(len(isterminal)) if isterminal[i] == 0]
    
    for j in range(int(sfa_batch_size/2)):        
        riffled[j*2, :, :, :] = s1[not_terminal_indices[j], :, :, :]
        riffled[j*2+1, :, :, :]= s2[not_terminal_indices[j], :, :, :]
    return riffled #s1, a, s2, isterminal, r


# In[ ]:

if __name__ == '__main__':
    # Create Doom instance
    game = initialize_vizdoom(config_file_path)

    # Action = which buttons are pressed
    n = game.get_available_buttons_size()
#    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    actions  = np.identity(n,dtype=int).tolist()

    # Create replay memory which will store the transitions
    memory = ReplayMemory(capacity=replay_memory_size)

    session = tf.Session()
    keras_controller = vizdoom_big_cnn(resolution[0], resolution[1], len(actions), lr=learning_rate, beta_1=0.95, beta_2=0.999)
    
    saver = tf.train.Saver()
    if load_model:
        print("Loading model from: ", model_savefile)
        saver.restore(session, model_savefile)
    else:
        init = tf.global_variables_initializer()
        session.run(init)
    print("Starting the training!")
    
    # "Manually" adapt learning rate
    epoch = 0
    def step_decay(keras_epoch):
        return (epochs-epoch) * 0.002 / float(epoch)
    lrate = LearningRateScheduler(step_decay)
    

    time_start = time()
    #net = sfa_vizdoom_default_cnn(resolution[0], resolution[1], 7) #general_sfa_net(resolution[0], resolution[1])
    if not skip_learning:
        for epoch in range(epochs):
            print("\nEpoch %d\n-------" % (epoch + 1))
            train_episodes_finished = 0
            train_scores = []

            print("Training...")
            game.new_episode()
            for learning_step in trange(learning_steps_per_epoch, leave=False):
                perform_learning_step(epoch)
                if game.is_episode_finished():
                    score = game.get_total_reward()
                    train_scores.append(score)
                    game.new_episode()
                    train_episodes_finished += 1

            print("%d training episodes played." % train_episodes_finished)

            train_scores = np.array(train_scores)

            print("Results: mean: %.1f±%.1f," % (train_scores.mean(), train_scores.std()),                   "min: %.1f," % train_scores.min(), "max: %.1f," % train_scores.max())

            print("\nTesting...")
            test_episode = []
            test_scores = []
            for test_episode in trange(test_episodes_per_epoch, leave=False):
                game.new_episode()
                while not game.is_episode_finished():
                    state = preprocess(game.get_state().screen_buffer)
#                    best_action_index = get_best_action(state)
                    best_action_index = np.argmax(keras_controller.predict(np.reshape(state, (1, state.shape[0], state.shape[1], 1))))

                    game.make_action(actions[best_action_index], frame_repeat)
                r = game.get_total_reward()
                test_scores.append(r)

            test_scores = np.array(test_scores)
            print("Results: mean: %.1f±%.1f," % (
                test_scores.mean(), test_scores.std()), "min: %.1f" % test_scores.min(),
                  "max: %.1f" % test_scores.max())

            print("Saving the network weights to:", model_savefile)
            saver.save(session, model_savefile)

            print("Total elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))

    game.close()
    print("======================================")
    print("Training finished. It's time to watch!")

    # Reinitialize the game with window visible
#    game.set_window_visible(True)
    game.set_mode(Mode.ASYNC_PLAYER)
    game.init()

    for _ in range(episodes_to_watch):
        game.new_episode()
        while not game.is_episode_finished():
            state = preprocess(game.get_state().screen_buffer)
            #best_action_index = get_best_action(state)
            best_action_index = np.argmax(keras_controller.predict(np.reshape(state, (1, state.shape[0], state.shape[1], 1))))


            # Instead of make_action(a, frame_repeat) in order to make the animation smooth
            game.set_action(actions[best_action_index])
            for _ in range(frame_repeat):
                game.advance_action()

        # Sleep between episodes
#        sleep(1.0)
        score = game.get_total_reward()
        print("Total score: ", score)



