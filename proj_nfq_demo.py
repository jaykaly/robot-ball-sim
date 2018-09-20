##############################
# proj_nfq_demo.py
#
# runs demo of ball experiment
# using NFQ algorithm
# input id(1-3) for 
# different goal configurations 
##############################

import sys
from math import pi, cos, sin
import numpy as np
from matplotlib import pyplot as plt

from robot_sim import *
from ball_sim import *
from reinf_learn_nfq import *
import pickle

##############################
### Robot parameters
### 6-dof
##############################

# link lengths
l = np.array((1,1,1,1,1,1),dtype=float)
l = np.array((1,1,0.5,0.5,0.5,0.5),dtype=float)

# Denavit-Hartenberg parameters for each joint
dh1 = [l[0],0,0,0.5*pi]
dh2 = [0,0,l[1],0]
dh3 = [0,0,l[2],0]
dh4 = [0,0,l[3],0]
dh5 = [0,0,l[4],0]
dh6 = [0,0,l[5],0]
DH_test = np.array((dh1,dh2,dh3,dh4,dh5,dh6),dtype=float)

##############################
        
def run_demo (exp_id=1):

    # Robot
    r = RobotSim()
    r.set_DH_params(DH_test)
    n_state_vars = 2*r.n_joints+1
    q_init = np.array(([0,0,0,0,0,0]))

    # Select experiment

    if exp_id == 1:
        # Experiment 1
        # Goal configuration
        goal = np.array(([-4,0,4]))
        radius = 1

        # Parameters
        max_len = 6
        n_iter = 10
        actions = get_actions_test
        train_wgt_file = './rob1_wgts_g303.h5'

    elif exp_id == 2:
        # Experiment 2
        # Goal configuration
        goal = np.array(([-5,0,3.5]))
        radius = 1

        # Parameters
        max_len = 6
        n_iter = 10
        actions = get_actions_test
        train_wgt_file = './rob1_wgts_g303.h5'

    elif exp_id == 3:
        # Experiment 3
        # Goal configuration
        goal = np.array(([-3,2,3.5]))
        radius = 1.5

        # Parameters
        max_len = 8
        n_iter = 12
        actions = get_actions_test2
        train_wgt_file = './rob1_wgts_g303.h5'

    else:
        print "Invalid experiment id"
        return

    print "Goal configuration:  "
    print "    Location:  ", goal[0],goal[1],goal[2]
    print "    Radius:    ", radius

    # Create model
    nn_model = build_nn_model (n_state_vars,wgt_file=train_wgt_file)
    
    # Reinforcement Learning algorithm
    a,rel = rl_nfq (r, goal, radius,
                    nn_model, q_init, v_dev=(0.025*pi), exp_prob=0.1, lbd=0.9,
                    action_func=actions,
                    max_len=max_len, n_iter=n_iter, n_trials=100)
    raw_input("Training complete, press any key to continue..")

    # Display final path
    print "Action sequence:"
    print a
    # Animation
    [q,v,rel,rwd,score] = run_single_trial (r,goal,radius,a,rel,q_init=q_init,
                                            animate=True,asize=5)
    print "Reward:  %1.4f  Score:  %1.2f" % (rwd,score)

    return

def main (id):
    exp_id = int(id)
    run_demo(exp_id)
    exit()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print "Usage:  proj_nfq_demo.py <exp_id>"
        exit()
