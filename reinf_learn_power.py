##############################
### reinforcement learning based on
### Policy Learning by Weighting Exploration with Returns
### PoWER method
###
### applied to ball experiment
###
### Reference:
###     J. Kober and J. Peters.  "Policy Search for Motor Primitives 
###     in Robotics".  Mach.Learn., July 2011.
###
##############################

from math import pi, exp
import numpy as np
import itertools

from robot_sim import *
from ball_sim import *

##############################

# Distance functions (phi)

def dist_norm (s,ref_pts):
    phi = np.zeros((len(ref_pts),1))
    for j,p in enumerate(ref_pts):
        phi[j] = np.linalg.norm(s-p)
    return phi

def dist_rbf (s,ref_pts,gamma=0.01):
    phi = np.zeros((len(ref_pts),1))
    norm = dist_norm(s,ref_pts)
    for j,p in enumerate(ref_pts):
        phi[j] = exp(-gamma*norm[j])
    return phi

# Returns a set of reference points
# input to phi function

def gen_ref_points ():
    ref_pts = []

    for j0 in [0.]:
        for j1 in [-0.25*pi,0.,0.10*pi,0.25*pi,0.50*pi]:
            for j2 in [-0.2*pi,0.,0.1*pi,0.2*pi,0.4*pi]:
                for j3 in [-0.2*pi,0.,0.1*pi,0.2*pi,0.4*pi]:
                    for j4 in [0.]:
                        for j5 in [0.]:
                            temp = np.array((j0,j1,j2,j3,j4,j5))
                            ref_pts.append(temp)
    for j0 in [-0.25*pi,0.25*pi]:
        for j1 in [-0.25*pi,0.,0.10*pi,0.25*pi,0.50*pi]:
            temp = np.array((j0,j1,0.,0.,0.,0.))
            ref_pts.append(temp)
                
    return ref_pts

# Create a training observation
# based on a single trial run

def gen_train_obs (q,v,rel,rwd,lbd=0.9):
    x_obs = q 
    # discounted reward
    y_obs = np.tile(1.,(len(x_obs),1))
    for k in range(len(y_obs)):
        y_obs[k] *= lbd**k
    y_obs = float(rwd)*np.flip(y_obs,0)

    return x_obs, y_obs

# Update policy
# with new data

def update_policy (policy, q_data, eps_data):
    for j in range(policy.shape[0]):
        r1 = 0.; r2 = 0.
        for k in range(len(q_data)):
            r1 += (eps_data[k][j] * q_data[k])
            r2 += q_data[k]
        r = r1 / r2
        policy[j,:] = policy[j,:] + r
    return policy

# Compute optimal path
# for given policy
    
def build_path_power (policy, ref_pts, q_init, 
                      phi_func=dist_rbf, max_len=5):

    a = np.zeros((max_len,policy.shape[0]))
    rel = np.flip(np.array((range(max_len))),0)

    q_curr = q_init
    v_curr = np.array(([0,0,0,0,0,0]))
        
    for k in range(max_len):
        act = policy.dot(phi_func(q_curr,ref_pts)).T
        q_curr = q_curr + v_curr
        v_curr = v_curr + act
        a[k,:] = act

    return a,rel

# Create sample trajectory
# using current policy
# and specified variance

def gen_sample_traj (robot, policy, ref_pts, 
                     q_init, sig,
                     phi_func=dist_rbf, max_len=5):

    a = np.zeros((max_len,robot.n_joints))
    rel = np.flip(np.array((range(max_len))),0)
    eps = np.zeros((max_len,robot.n_joints,len(ref_pts)))

    q_curr = q_init
    v_curr = np.array(([0,0,0,0,0,0]))
        
    for k in range(max_len):
        eps_test = np.random.normal(0,sig,size=policy.shape)
        policy_test = policy + eps_test
        act_test = policy_test.dot(phi_func(q_curr,ref_pts)).T

        q_curr = q_curr + v_curr
        v_curr = v_curr + act_test
        a[k,:] = act_test
        eps[k,:,:] = eps_test

    return [(a,rel),eps]

# Create data for updating policy
# samples trajecries with current policy
# and returns Q-estimates for updating policy

def gen_sample_data (robot, goal, radius,
                     policy, ref_pts, 
                     q_init, sig=0.05, lbd=0.9,
                     phi_func=dist_rbf,
                     max_len=5, n_trials=50):

    q_data = np.zeros((0,1))
    eps_data = np.zeros((0,robot.n_joints,len(ref_pts)))
    rwd_hist = []
    score_hist = []

    for j in range(n_trials):
        # generate sample trajectory
        (test_a,test_rel),test_eps = gen_sample_traj(robot, policy, ref_pts,
                                                     q_init, sig=sig,
                                                     phi_func=phi_func,
                                                     max_len=max_len)

        # run experiment and add result to dataset
        (q,v,rel,rwd,score) = run_single_trial (robot,goal,radius,test_a,test_rel)
        x_temp,y_temp = gen_train_obs(q,v,rel,rwd,lbd=lbd)
        q_data = np.vstack((q_data,y_temp))
        eps_data = np.vstack((eps_data,test_eps))

        # store reward and score, for reporting
        rwd_hist.append(rwd)
        score_hist.append(score)

    avg_rwd = np.mean(np.array((rwd_hist)))
    avg_score = np.mean(np.array((score_hist)))

    return [q_data,eps_data],[avg_rwd,avg_score]

# Runs PoWER algorithm
# multiple iterations of updating policy parameters
# computes average reward/score over trials

def rl_power (robot, goal, radius,
              q_init, ref_pts_func=gen_ref_points,
              policy_init=[], sig=0.05, lbd=0.9,
              max_len=5, n_iter=10, n_trials=50, save_file=[]):

    # set of reference points
    ref_pts = ref_pts_func()

    # initialize policy
    policy = np.zeros((robot.n_joints,len(ref_pts)))
    if len(policy_init):
        policy = policy_init

    result = np.zeros((n_iter,2))
    max_score = 0.

    print "Running PoWER algorithm"
    for j in range(n_iter):

        # sample trajectories using current policy
        # and compute Q-estimates
        [q_data,eps_data],[rwd,score] = gen_sample_data (robot, goal, radius,
                                      policy, ref_pts,
                                      q_init, sig=sig, lbd=lbd,
                                      max_len=max_len, n_trials=n_trials)       

        result[j][0] = rwd
        result[j][1] = score
        print "Iter %d:  %1.4f  %1.2f" % (j+1,rwd,score)

        # update policy using the new data samples
        policy = update_policy(policy,q_data,eps_data)

        if score >= max_score:
            # store optimal action sequence from best model            
            a,rel = build_path_power (policy,ref_pts,q_init,
                                      max_len=max_len)
            max_score = score

            # Speed convergence
            sig *= 0.99
            if max_score >= 0.25:
                sig *= 0.95 #0.99
            if max_score >= 0.50:
                sig *= 0.95 #0.98
            
    if len(save_file):        
        with open(save_file,'w') as fp:
            pickle.dump(policy,fp)
        print "weights saved to:  ",save_file
    print "Complete"

    return a,rel






