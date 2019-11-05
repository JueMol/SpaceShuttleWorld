import math
import matplotlib.pyplot as plt
from Physics.SpaceShuttlePhysics import SpaceShuttlePhysics

''' Test the SpaceShuttlePhysics class by running an episode with the 
    shuttle's built in naive action policy.
    
    Draw the result
    
    The shuttle is denoted as a triangle
    The rocket thrust as a red vector at its back
    The trajectory is drawn as a (light) blue line 
    The worlds boundary as a solid blue line
'''

#
# Calculate and plot the solution trajectory by the naive autopilot
#
envi_demo = SpaceShuttlePhysics(runtime=100.)
envi_demo.do_demo()


