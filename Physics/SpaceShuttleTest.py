import math
import matplotlib.pyplot as plt

''' Test the SpaceShuttlePhysics class by running an episode with the 
    shuttle's built in naive action policy.
    
    Draw the result
    
    The shuttle is denoted as a triangle
    The rocket thrust as a red vector at its back
    The trajectory is drawn as a dotted line 
    The worlds boundary as a solid blue line
'''

from SpaceShuttlePhysics import SpaceShuttlePhysics

shuttle_sim = SpaceShuttlePhysics()
shuttle_sim.demo()

