##############################
### Robot Learning seminar
###
### robot simulation python class
### implements forward kinematics
### and simple animation
##############################

from math import pi, cos, sin
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import animation as animation

##############################

##############################
# Transformation matrices
##############################

# Transform matrix using DH params
# from frame (n-1) to (n)
# = Z_mat*X_mat
def t_mat (theta,d,alpha,a):
    ct = cos(theta); st = sin(theta)
    ca = cos(alpha); sa = sin(alpha)
    return np.array(([ct, -st*ca, st*sa,  a*ct], \
                     [st, ct*ca,  -ct*sa, a*st], \
                     [0,  sa,     ca,     d], \
                     [0,  0,      0,      1]),dtype=float)


##############################
# Plots and animation
##############################

# Plot object in 3-D
def plot_3d_simple (pts,edges=[],asize=1,text=[]):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    if len(text):
        ax.text2D(0,0,text)

    # If specified, connect points
    if (edges == 1):
        edges = [(j,j+1) for j in range(len(pts)-1)]    

    # Axes
    ax.plot([0,asize],[0,0],zs=[0,0],color='gray',linestyle='--')
    ax.plot([0,0],[0,asize],zs=[0,0],color='gray',linestyle='--')
    ax.plot([0,0],[0,0],zs=[0,asize],color='gray',linestyle='--')
    # Points
    for (x,y,z) in pts:
        ax.plot([x],[y],zs=[z],marker='o',color='red')
    # Links
    for (v1,v2) in edges:
        plt.plot([pts[v1][0],pts[v2][0]],[pts[v1][1],pts[v2][1]], \
                 [pts[v1][2],pts[v2][2]],color='blue')
    plt.show()    

# Robot (manipulator) animation (3-D)
def anim_3d_simple (ts_pts,connect=True,asize=10,text=[]):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # axes
    ax.plot([0,asize],[0,0],zs=[0,0],color='gray',linestyle='--')
    ax.plot([0,0],[0,asize],zs=[0,0],color='gray',linestyle='--')
    ax.plot([0,0],[0,0],zs=[0,asize],color='gray',linestyle='--')

    _pts,  = ax.plot([],[],zs=[],marker='o',color='red',linestyle='None')
    _line, = ax.plot([],[],zs=[],color='blue')
    _text = ax.text2D(0,0,'')
    
    for t in range(len(ts_pts)):
        x = ts_pts[t][:,0]
        y = ts_pts[t][:,1]
        z = ts_pts[t][:,2]
        _pts.set_data(x,y)
        _pts.set_3d_properties(z)
        if connect:
            _line.set_data(x,y)
            _line.set_3d_properties(z)
        if len(text):
            _text.set_text(text[t])
        plt.pause(0.5)


##############################
# RobotSim class
##############################

class RobotSim(object) :
    ''' class for simulating robot kinematics '''
    
    def __init__(self):
        self.initialized = False

        self.n_joints = 0
        # Denavit-Hartenberg parameters for each joint
        self.DH = np.array((),dtype=float)
        # rannge of motion for each joint
        self.q_min = np.array((),dtype=float)
        self.q_max = np.array((),dtype=float)
        # current joint position
        self.q = np.array((),dtype=float)

    def set_DH_params(self,DH):
        self.DH = DH
        self.n_joints = len(DH)

        # range of motion for each joint
        # default, not currently used
        self.q_min = np.tile(0.,(1,self.n_joints))
        self.q_max = np.tile(2.*pi,(1,self.n_joints))
        # current joint position
        self.q = np.tile(0.,(1,self.n_joints))

        self.initialized = True

    def set_motion_range(self,q_min=[],q_max=[]):
        self.q_min = q_min
        self.q_max = q_max
        
    # fwd_kinematics
    # Inputs:   sequence joint angles (q1,q2,...,qn)
    # Outputs:  location of joints/end-effector (x0,x1,x2,...,xn)
    #           relative to origin
    def fwd_kinematics (self,q):
        x = np.zeros((q.shape[0]+1,4))
        x[:,3] = 1 # homogenous coords

        t_mats = [[] for joint in self.DH]
        for j in range(len(q)):
            t_mats[j] = t_mat(q[j]+self.DH[j][1],
                              self.DH[j][0],
                              self.DH[j][3],
                              self.DH[j][2])

        tr_mats = [np.eye(4) for joint in self.DH]
        for j in range(len(tr_mats)):
            for k in range(j,len(tr_mats)):
                tr_mats[k] = tr_mats[k].dot(t_mats[j])
        for j in range(len(tr_mats)):
            x[j+1,:] = tr_mats[j].dot(x[j+1,:])    
        return x[:,0:3]

    def plot_3d (self,q,input='q',asize=1,text=[]):
        x = self.fwd_kinematics(q)
        plot_3d_simple(x,edges=1,asize=asize,text=text)

    def animate_3d (self,q_ts,input='q',asize=10,text=[]):
        x_ts = []
        for q in q_ts:
            x = self.fwd_kinematics(q)
            x_ts.append(x)
        anim_3d_simple(x_ts,connect=True,asize=asize,text=text)

 

##############################
# Run animation given sequence of angles
##############################

DH_test = []
def animate_robot (q_list,DH=DH_test,asize=5,text=[]):
    joints_ts = []
    for q in q_list:
        x = fwd_kinematics(q,DH)
        joints_ts.append(x)
    anim_3d_simple(joints_ts,connect=True,asize=asize,text=text)

##############################

