##############################
# proj_ball_demo.py
#
# displays animations of
# ball experiment
# for different configurations 
##############################

from math import pi, cos, sin
import numpy as np
from matplotlib import pyplot as plt

from robot_sim import *
from ball_sim import *

##############################
### Robot parameters
### 6-dof
##############################

# link lengths
l = np.array((1,1,1,1,1,1),dtype=float)
l = np.array((1,1,0.5,0.5,0.5,0.5),dtype=float)

# Denavit-Hartenberg parameters for each joint
# for each joint specify:
#     d: distance along axis z(n-1)
#     theta: rotation around z(n-1) to reach new x
#            adjusted by input parameter q
#     a: distance along axis x(n)
#     alpha: rotation around x(n) to reach new z
dh1 = [l[0],0,0,0.5*pi]
dh2 = [0,0,l[1],0]
dh3 = [0,0,l[2],0]
dh4 = [0,0,l[3],0]
dh5 = [0,0,l[4],0]
dh6 = [0,0,l[5],0]
DH_test = np.array((dh1,dh2,dh3,dh4,dh5,dh6),dtype=float)

# range of motion for each joint
# not currently used
q_min = np.tile(0.,(1,6))
q_max = np.tile(2.*pi,(1,6))

##############################
        
def main ():

    # Robot
    r = RobotSim()
    r.set_DH_params(DH_test)

    # Example 1

    goal = np.array(([-3,0,3]))
    radius = 1

    a = np.tile(np.array(([0,0.1*pi,0.05*pi,0,0.05*pi,0.05*pi])),(5,1))
    rel = np.flip(np.array((range(5))),0)
    run_single_trial (r,goal,radius,a,rel,animate=True,asize=5)

    # Example 2

    goal = np.array(([-4,0,4]))
    radius = 2

    a = np.tile(np.array(([0,0.1*pi,0.07*pi,0,0.05*pi,0.03*pi])),(6,1))
    rel = np.flip(np.array((range(6))),0)
    run_single_trial (r,goal,radius,a,rel,animate=True,asize=5)

    # Example 3

    goal = np.array(([-3,2,3]))
    radius = 2

    a1 = np.tile(np.array(([-0.1*pi,0,0,0,0,0])),(4,1))
    a2 = np.tile(np.array(([0.05*pi,0.1*pi,0.05*pi,0,0.05*pi,0.05*pi])),(5,1))
    a = np.vstack((a1,a2))
    rel = np.flip(np.array((range(9))),0)
    run_single_trial (r,goal,radius,a,rel,animate=True,asize=5)

    exit()

if __name__ == '__main__':
    main()
