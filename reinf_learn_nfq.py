##############################
### reinforcement learning based on
### Neural-network based Q-estimation
###
### applied to ball experiment
###
### Reference:
###     M. Riedmiller. "Neural Fitted Q iteration: First experiences 
###     with a data efficient neural reinforcement learning method".
###     16th European Conference on Machine Learning, 2005.
###
##############################

import numpy as np
import itertools

from robot_sim import *
from ball_sim import *

##############################

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam

##############################

# Create a NN training observation
# based on a single trial run

def gen_train_obs (q,v,rel,rwd,lbd=0.9):
    if len(rel.shape)==1:
        rel = np.expand_dims(rel,1)

    x_obs = np.hstack((q,v,rel))
    # discounted reward
    y_obs = np.tile(1.,(len(x_obs),1))
    for k in range(len(y_obs)):
        y_obs[k] *= lbd**k
    y_obs = float(rwd)*np.flip(y_obs,0)

    return x_obs, y_obs

# Create training set for initializing NN
# based on "imitation" of known good paths
# along with random component

def gen_data_from_example (robot, goal, radius, ex_a, ex_rel,
                           q_init=[],
                           n_trials=50, v_dev=(0.05*pi), lbd=0.9):

    n_steps = len(ex_a)
    x_data = np.zeros((0,2*robot.n_joints+1))  # state: pos+vel+time
    y_data = np.zeros((0,1))                   # reward

    # run trials using small deviations from example path
    for j in range(n_trials):
        # generate path
        test_a = ex_a + np.random.randint(-2,3,size=(n_steps,robot.n_joints))*v_dev
        test_rel = np.flip(np.array((range(n_steps))),0)
        
        # run experiment and add result to dataset
        (q,v,rel,rwd,score) = run_single_trial (robot,goal,radius,test_a,test_rel,q_init=q_init)
        x_temp,y_temp = gen_train_obs(q,v,rel,rwd,lbd=lbd)
        x_data = np.vstack((x_data,x_temp))
        y_data = np.vstack((y_data,y_temp))

    return [x_data,y_data]

# Define a neural network in Keras
# and load/train weights if specified

def build_nn_model (n_inputs,n_hidden=24,wgt_file=[],seed_data=[]):
    model = Sequential()
    model.add(Dense(n_hidden, input_shape=(n_inputs,), activation='relu'))
#    model.add(Dropout(0.2))
    model.add(Dense(n_hidden, activation='relu'))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    print "Created model"
    #print model.summary()

    if len(wgt_file):
        model.load_weights(wgt_file)
        print "Loaded weights from file: ",wgt_file
    if len(seed_data):
        x_data,y_data = seed_data
        model.fit (x_data,y_data,batch_size=32,epochs=20,verbose=1)
        print "Trained on seed data, 20 epochs"
    return model

# Define a limited set of actions
# discrete values for acceleration
# for each joint and time step

def get_actions_test (v_dev=(0.025*pi)):
    actions = np.zeros((0,6))
    ja = [-1*v_dev,0,1*v_dev,2*v_dev,4*v_dev]

    for j0 in [0.]:
        for j1 in ja:
            for j2 in ja:
                for j3 in ja:
                    for j4 in [-v_dev,0.,v_dev]:
                        for j5 in [0.]:
                            temp = np.array((j0,j1,j2,j3,j4,j5))
                            actions = np.vstack((actions,temp))    
    return actions

def get_actions_test2 (v_dev=(0.025*pi)):
    actions = np.zeros((0,6))
    ja = [-1*v_dev,0,1*v_dev,2*v_dev,4*v_dev]

    for j0 in [-2*v_dev,0,4*v_dev]:
        for j1 in ja:
            for j2 in ja:
                for j3 in ja:
                    for j4 in [0.]:
                        for j5 in [0.]:
                            temp = np.array((j0,j1,j2,j3,j4,j5))
                            actions = np.vstack((actions,temp))    
    return actions

# Build a sequence of actions to follow
# using the specified NN model/policy
# exp_prob = 0 will return optimal actions with current Q-function
# positive values specifiy probability of random exploration
    
def build_path_nn (model,q_init,action_func=get_actions_test,v_dev=(0.05*pi),
                   exp_prob=0.,max_len=10):
    q_curr = q_init
    v_curr = np.array(([0,0,0,0,0,0]))
    rel_curr = max_len
    state = np.hstack((q_curr,v_curr,np.array((rel_curr))))

    # Get list of possible actions
    # (hard-coded for 6-dof robot, to limit state space for testing
    act_list = action_func (v_dev=v_dev)
    
    path_a   = np.zeros((0,q_curr.shape[0]))
    path_rel = np.zeros((0,1))

    path_len = 0
    released = False
    while path_len < max_len and not released:
        # test possible next actions
        test_states = np.zeros((0,state.shape[0]))
        q_next = q_curr + v_curr
        rel_next = rel_curr - 1
        for act in act_list:
            v_next = v_curr + act
            test_state = np.hstack((q_next,v_next,np.array((rel_next))))
            test_states = np.vstack((test_states,test_state))

        test_Q = model.predict(test_states).flatten()
        max_idx = np.where(test_Q==np.max(test_Q))
        best_acts = act_list[max_idx]

        # select next action 
        z = np.random.random()
        if z < exp_prob:
            # random selection (exploration)
            next_idx = np.random.choice(len(act_list))
            next_act = act_list[next_idx]
        else:
            # select from best actions (exploitation)
            next_idx = np.random.choice(len(best_acts))
            next_act = best_acts[next_idx]

        # update path and state
        path_a = np.vstack((path_a,next_act))
        path_rel = np.vstack((path_rel,np.array((rel_next))))

        if rel_next == 0:
            released = True
        q_curr = q_curr + v_curr
        v_curr = v_curr + next_act
        rel_curr = rel_next
        path_len += 1

    path_rel = path_rel.flatten()

    return path_a,path_rel

# Create data for training NN
# using current NN-based Q-function
# Samples paths and computes score/reward

def gen_data_nn (robot, goal, radius, 
                 model, q_init, action_func=get_actions_test, v_dev=(0.05*pi), 
                 exp_prob=0.2, lbd=0.9,
                 max_len=10, n_trials=50):

    x_data = np.zeros((0,2*robot.n_joints+1))  # state: pos+vel+release
    y_data = np.zeros((0,1))                   # reward
    rwd_hist = []
    score_hist = []
    
    for j in range(n_trials):
        # generate path
        test_a,test_rel = build_path_nn (model,q_init,
                                         action_func=action_func,
                                         v_dev=v_dev,exp_prob=exp_prob,
                                         max_len=max_len)
        
        # run experiment and add result to dataset
        (q,v,rel,rwd,score) = run_single_trial (robot,goal,radius,test_a,test_rel,q_init=q_init)
        x_temp,y_temp = gen_train_obs(q,v,rel,rwd,lbd=lbd)
        x_data = np.vstack((x_data,x_temp))
        y_data = np.vstack((y_data,y_temp))
        # store reward and score, for reporting
        rwd_hist.append(rwd)
        score_hist.append(score)

    avg_rwd = np.mean(np.array((rwd_hist)))
    avg_score = np.mean(np.array((score_hist)))

    return [x_data,y_data],[avg_rwd,avg_score]

# Runs Neural-Fitted Q-iteration algorithm
# multiple iterations of generating training samples
# and updating the NN-model
# computes average reward/score over trials

def rl_nfq (robot, goal, radius,
            model, q_init, action_func=get_actions_test, v_dev=(0.05*pi), 
            exp_prob=0.2, lbd=0.9,
            max_len=10, n_iter=10, n_trials=50, save_file=[]):

    result = np.zeros((n_iter,2))
    max_score = 0.

    print "Running NFQ algorithm"
    for j in range(n_iter):
        [x_data,y_data],[rwd,score] = gen_data_nn (robot, goal, radius, 
                                     model, q_init, action_func=action_func, 
                                                   v_dev=v_dev, 
                                     exp_prob=exp_prob, lbd=lbd,
                                     max_len=max_len, n_trials=n_trials)
        result[j][0] = rwd
        result[j][1] = score
        print "Iter %d:  %1.4f  %1.2f" % (j+1,rwd,score)
        
        if score >= max_score:
            # store optimal action sequence from best model            
            a,rel = build_path_nn (model,q_init,
                                   action_func=action_func,
                                   v_dev=v_dev,exp_prob=0.,
                                   max_len=max_len)
            max_score = score
            # Speed convergence
            if max_score > 0.50:
                exp_prob = 0.95*exp_prob
            if max_score > 0.80:
                exp_prob = 0.95*exp_prob

        # update NN weights using the new data samples
        model.fit (x_data,y_data,batch_size=32,epochs=3,verbose=0)

    if len(save_file):
        model.save_weights(save_file)
        print "weights saved to:  ",save_file
    print "Complete"

    return a,rel







