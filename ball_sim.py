##############################
### functions to implement
### ball/hoop experiment
###
### compute trajectories, score, reward
### animations
##############################

from math import pi, cos, sin
import numpy as np
from matplotlib import pyplot as plt

from robot_sim import *
from numpy import genfromtxt

##############################

grav = 9.8

##############################

# Compute trajectory of ball given initial velocity

def ball_trajectory (x_init, v_init, n_steps=20, t_step_size=0.1):
    traj = np.zeros((n_steps+1,3))
    traj[0,:] = x_init
    v = v_init
    a = np.array((0,0,-grav))
    for t in range(1,n_steps+1):
        v = v + t_step_size*a
        traj[t,:] = traj[t-1,:] + t_step_size*v
    return traj

# Find intersection of ball with goal
# on downward trajectory

def z_intersect (traj, z_val=0):
    res = []
    p = 1
    while not len(res) and p < len(traj):
        if traj[p-1,2] > z_val and traj[p,2] <= z_val:
            # Interpolate
            temp = (traj[p-1,2] - z_val) / (traj[p-1,2] - traj[p,2]) 
            x_int = (1-temp) * traj[p-1,0] + temp * traj[p,0]
            y_int = (1-temp) * traj[p-1,1] + temp * traj[p,1]
            res = np.array((x_int,y_int,z_val))
        p += 1
    return res

# Score and Reward functions

def compute_score (traj, goal, radius):
    z_int = z_intersect(traj,goal[2])
    score = 0.
    if len(z_int):
        dist = np.linalg.norm(z_int-goal)
        if dist < radius:
            score = 1.
    return score

def dist_reward (traj, goal, radius):
    z_int = z_intersect(traj,goal[2])
    rwd = 0.
    if len(z_int):
        dist = np.linalg.norm(z_int-goal)
        rwd = max(0,1.*(radius-dist)/(radius))
    return rwd

def height_reward (traj, goal, radius):
    max_h = np.max(traj[:,2])
    rwd = 0.
    if max_h > 0:
        rwd = min(1.,max_h/(goal[2]+radius))
    return rwd

def direction_reward (traj, goal, radius):
    rwd = 0.
    x_dir = (traj[1][0] - traj[0][0]) / (goal[0] - traj[0][0])
    y_dir = (traj[1][1] - traj[0][1]) / (goal[1] - traj[0][1])
    if x_dir > 0 and y_dir > 0:
        rwd = 1.
    return rwd

def combined_reward (traj, goal, radius):    
    # Component 0:  score
    score = compute_score (traj,goal,radius)
    # Component 1:  distance from goal during downward trajectory
    rwd1 = dist_reward (traj,goal,5.)
    # Component 2:  partial reward for reaching height close to goal
    rwd2 = height_reward (traj,goal,radius)
    # Component 3:  partial reward for moving in correct (x,y) direction
    rwd3 = direction_reward (traj,goal,radius)
    # Weights
    w0,w1,w2,w3 = [0.45,0.45,0.05,0.05]
    rwd = w0*score + w1*rwd1 + w2*rwd2 + w3*rwd3
    return rwd

# Run a single trial
# given a time series of actions (accelerations)
# compute score and reward
# show animation (optional)

def run_single_trial (robot, goal, radius, v_del, rel, 
                      q_init=[],
                      rwd_func=combined_reward, t_step_size=0.1, 
                      animate=False, asize=10):
    n_steps = len(v_del)
    x_joints = []                             # position of all joints
    x = np.zeros((n_steps+1,3))               # position of end-effector
    q = np.zeros((n_steps+1,len(v_del[0])))   # joint angles
    v = np.cumsum(t_step_size*v_del,axis=0)   # joint velocity
    q[1:,:] = q[1:,:] + np.cumsum(v,axis=0)

    # compute joint and ee locations
    for t in range(n_steps+1):
        fk = robot.fwd_kinematics(q[t,:])
        x_joints.append(fk)
        x[t,:] = fk[-1]

    # find release point
    temp = np.where(rel==0)[0]
    if len(temp):
        t_rel = temp[0]+1
    else:
        t_rel = -1

    # velocity of end-eff in x-y-z space
    v_ee = (x[t_rel,:] - x[t_rel-1,:]) / t_step_size
    # ball trajectory
    traj = ball_trajectory(x[t_rel,:],v_ee,n_steps=20,t_step_size=t_step_size)

    score = compute_score(traj,goal,radius)
    rwd = rwd_func(traj,goal,radius)
 
    # plot
    if animate:
        z_int = z_intersect(traj,goal[2])
        animate_exp (np.array((x_joints)),traj,z_int,rwd,score,
                     goal,radius,asize=asize)

    # return state variables up to release point, and final score
    q = q[:t_rel,:]
    v = v[:t_rel,:]
    rel = rel[:t_rel]

    return [q,v,rel,rwd,score]

####################

# Experiment animation (3-D)

pt_short = 0.12
pt_long  = 2.00

def animate_exp (ts_joints,ts_ball,result,rwd,score,goal,radius,asize=10):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # axes
    ax.plot([0,asize],[0,0],zs=[0,0],color='gray',linestyle='--')
    ax.plot([0,0],[0,asize],zs=[0,0],color='gray',linestyle='--')
    ax.plot([0,0],[0,0],zs=[0,asize],color='gray',linestyle='--')

    # goal
    ax.plot([goal[0]],[goal[1]],zs=[goal[2]],marker='+',color='green',linestyle='None')
    x = [goal[0],goal[0]+radius,goal[0],goal[0]-radius,goal[0]]
    y = [goal[1]-radius,goal[1],goal[1]+radius,goal[1],goal[1]-radius]
    z = [goal[2],goal[2],goal[2],goal[2],goal[2]]
    ax.plot(x,y,zs=z,color='green',linestyle='--')
    
    _joints, = ax.plot([],[],zs=[],marker='o',color='red',linestyle='None')
    _line,   = ax.plot([],[],zs=[],color='blue')
    _ball,   = ax.plot([],[],zs=[],marker='o',color='orange',linestyle='None')
    _text = ax.text2D(0,0,'')

    n1_steps = len(ts_joints)
    n2_steps = len(ts_ball)-1

    for t in range(n1_steps):
        x = ts_joints[t][:,0]
        y = ts_joints[t][:,1]
        z = ts_joints[t][:,2]
        _joints.set_data(x,y)
        _joints.set_3d_properties(z)
        _line.set_data(x,y)
        _line.set_3d_properties(z)
        _text.set_text('t = %d' % t)
        plt.pause(pt_short)
    for t in range(1,n2_steps):
        x = ts_ball[t,0]
        y = ts_ball[t,1]
        z = ts_ball[t,2]
        _ball.set_data(x,y)
        _ball.set_3d_properties(z)
        _text.set_text('t = %d' % (n1_steps+t))
        plt.pause(pt_short)
    if len(result):
        _ball.set_data(result[0],result[1])
        _ball.set_3d_properties(result[2])
    _text.set_text('Score: %d  Reward: %.3f' % (score,rwd))
    plt.pause(pt_long)
