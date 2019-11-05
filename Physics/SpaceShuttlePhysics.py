# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 19:09:30 2019

@author: Juergen Mollen (Git:JueMol)
"""

import math as math
import numpy as np
import matplotlib.pyplot as plt
    
class Trajectory():
    ''' Convenient class to help with remembering trajectories '''
    def __init__(self):
        self.posx = []
        self.posy = []
        self.velx = []
        self.vely = []
        self.accx = []
        self.accy = []
        self.angl = []
        self.t = 0.
        self.episode = None
    
    def remember_step(self, pos, vel, acc, ang, t):
        self.posx.append(pos[0])
        self.posy.append(pos[1])
        self.velx.append(vel[0])
        self.vely.append(vel[1])
        self.accx.append(acc[0])
        self.accy.append(acc[1])
        self.angl.append(ang)
        self.t = t
        
    def gettrajectory(self):
        return self.posx,\
            self.posy,\
            self.velx,\
            self.vely,\
            self.accx,\
            self.accy,\
            self.angl,\
            self.t
    

X = 0   # ... used to make the code below more readable 
Y = 1

class SpaceShuttlePhysics():
    
    '''
    Simulates the physics of a space shuttle that has a constant thrust in a 100x100 world
    
    Initial position is (x=10, y=0) 
    Initial heading of the shuttle is vertical 
    Initial velocity is zero
    The goal is, to reach the right world boundary within in the y range between 60 and 100
    There exist an obstacle as rectangle of height 60 positioned at x=60 to 70
    '''

    
    def __init__(self, runtime=100.):
        self.init_pose = np.array([10.0, 1.0])   # X=10, Y=1
        self.init_velo = np.array([0.0, 0.0])    # stand still
        self.init_angle = 90.0                   # start with vertical orientation
        self.pos = self.init_pose
        self.vel = self.init_velo
        self.ang = self.init_angle
        self.dt_ref = 0.2
        self.dt = 1.
        self.thrust_max = 0.2
        self.runtime = runtime
        self.demo = False
        self.obst_height = 55.                   # ... made the height of the obstacle variable
        
        # create action space
        tmp = []
        tmp.append(np.array([-1, self.thrust_max]))
        tmp.append(np.array([ 0, self.thrust_max]))
        tmp.append(np.array([ 1, self.thrust_max]))
        self.real_action_space = np.array(tmp)

        # Worlds boundaries (used for drawing)
        self.x_bound = [100, 0., 0., 60., 60., 70., 70., 100., 100.]  
        self.y_bound = [100., 100, 0., 0., self.obst_height, self.obst_height, 0., 0., 60.]
        
        self.bestpath = Trajectory()
        self.bestpath.t = np.inf
        self.bestpath_dist = np.inf
        
        self.dt_steps = [0.4, 0.5, 0.6, 0.8, 1.0]

        self.reset()
        
        

    ################################
    # Algorithmic helper functions #
    ################################
    
    def get_trust_vector(self, ang):
        return np.array([math.cos(ang*math.pi/180.0), math.sin(ang*math.pi/180.0)]) # ... think in 360 dregee angles 

    def get_real_action(self, action):
        return(self.real_action_space[action])

    def norm_vector(self, vec):
        return vec/self.len(vec)

    def len(self, vec1):
        return np.sqrt(np.sum(np.square(vec1)))

    def distance(self, vec1, vec2):
        return self.len(vec2-vec1)

    def scalar_product(self, vec1, vec2):
        return np.dot(vec1, vec2)

    def state_size(self):
        return 5

    def action_size(self):
        return len(self.real_action_space)

    def get_angle(self, vec):
        tmp_ang = math.atan2(vec[Y], vec[X]) * 180.0 / math.pi
        if tmp_ang < 0.0:
            tmp_ang += 360.0
        return tmp_ang

    def get_segment_fraction_in_area(self, pos_old, pos):
        segment_frac_in_area = 1.0
        segment_frac_Y = 1.0
        segment_frac_X = 1.0
        
        if pos_old[Y] < 100. and pos[Y] >= 100.:
            segment_frac_Y = (100.-pos_old[Y])/(pos[Y]-pos_old[Y])+0.01
        if pos_old[X] < 100. and pos[X] >= 100.:  
            segment_frac_X = (100.-pos_old[X])/(pos[X]-pos_old[X])+0.01
        
        segment_frac_in_area = min(segment_frac_Y, segment_frac_X)
        
        if pos_old[X] < 60. and pos[X] >= 60. and pos[Y] <= self.obst_height:
            segment_frac_in_area = (60.-pos_old[X])/(pos[X]-pos_old[X])+0.01
        return segment_frac_in_area

    #####################################################
    # Event funtions with respect to the world boundary #
    #####################################################

    def obstacle(self, pos):
        ''' Checks if the obstacle is hit
        (thereby the size of shuttle is not taken to account) '''
        
        return pos[X] >= 60. and pos[X] <= 70.0 and pos[Y] <= self.obst_height
        
    def crashes(self, pos):
        
        ''' Checks for collisions.
        (thereby the size of shuttle is not taken to account) '''
        
        if pos[X] < 0.0:                              # leaves world to the left
            return True
        elif pos[Y] > 100.0 or pos[Y] < 0.0:          # leaves world to top or bottom 
            return True
        elif pos[X] > 100.0 and pos[Y] < 60.0:        # Hits right wall
            return True
        
    def reached_goal(self, pos) :
        
        ''' Checks whether the goal ('docking bay') is reached.
        (thereby the size of shuttle is not taken to account) '''

        return pos[X] > 100.0 and pos[Y] > 60.0 and pos[Y] < 100.0  # leaves world to the right, but through docking bay entry


    def naive_policy_action(self):
        
        ''' Get Action from naive linear policy.
        (It works mainly such, that the steering tends to point toward the (tempory) goal) '''
        
        top_of_goal = np.array([100.0, 100.0]) - self.pos   # ... top of goal (docking bay)
        target_vector = np.array([100.0, 60.0]) - self.pos  # ... bottom of goal (docking bay)

        # Dont steer too early - just if just  about to see the docking bay otherwise steer to corner of obstacle
        if (self.pos[1] + (top_of_goal[Y]-self.pos[1])/top_of_goal[X] * (60.-self.pos[X])) < 60. and self.pos[Y] < 30.:
            target_vector = np.array([60., self.obst_height+5.]) - self.pos 
            
        self_ang = int(self.ang)
        heading_ang = int(self.get_angle(target_vector))

        diff_ang = heading_ang-self_ang
        if diff_ang > 180.:
            diff_ang -= 360.

        return int(np.sign(diff_ang)) + 1  # steer always in the direction of the targets
    
    
    ####################################
    # Functions used for Deep Learning #
    ####################################
        
    def reset(self, eps=None):
        
        ''' Resets the shuttles physics.
            The shuttle is (again) placed in the lower left corner, heading up wit zero velocity
        '''
        
        self.time = 0.0
        self.pos = np.copy(self.init_pose)
        self.vel = np.copy(self.init_velo)
        self.ang = self.init_angle   
        self.has_hit_obstacle = False

        self.path = Trajectory()
        self.path.episode = eps
        
        return self.get_state()

    def get_state(self):
        
        '''Returns the simulations state.
           This is an array (tuple) of:
               - position vector
               - Velocity Vector
               - and the steering angle with respect to the trajectory (seems to be better than absolute angle)        
        '''

        state = np.array([self.pos[X], self.pos[Y], self.vel[X], self.vel[Y], self.ang])
        return np.reshape(state, [1, self.state_size()])
    
    
    def reward(self):        

        '''Calculate the reward for each step.
           Penalize each step, by comparing passed distance with approach to mid of goal
           If the shuttle hits the boundary penalize with gradient (to the top/right)
           If the shuttle reaches goal give extra reward and penalise distance from mid of goal
        '''

        reward = 0.
        
        acc = self.thrust * self.get_trust_vector(self.ang)
        pos = np.copy(self.pos)
        vel = np.copy(self.vel)
        start_pos = np.copy(pos)
    
        path_len = 0.
        
        # Do litte lookahead (with constant current action) to calculate reward
        for i in range(4):
            pos_old = np.copy(pos)
            vel += acc * self.dt        # ... the only mechanics
            pos += vel * self.dt   

            # Calculate accurate segment if leaves world (this smoothes gradient)
            segment_frac_in_area = self.get_segment_fraction_in_area(pos_old, pos) 
            if segment_frac_in_area < 1.0:
                pos = pos_old + vel * self.dt * segment_frac_in_area 
                
            path_len += self.len(pos-pos_old)
            
            crash    = self.crashes(pos)
            goal     = self.reached_goal(pos)
            obstacle = self.obstacle(pos)

            if pos[X] < 1.:                                             #  hit left bound
                reward -= 200. - pos[Y]
            elif pos[Y]  > 99.:                                         #  hit upper bound
                reward -= (100.-pos[X])
            elif pos[X]  > 99.:                                         #  hit right bound
                reward += 40. - abs(pos[Y]-85)
            elif self.pos[X] > 59. and self.pos[Y] < self.obst_height:  #  hit obstacle
                reward -= 100. + 10*(self.obst_height-pos[Y])
            elif pos[Y] < 1.:                                           #  hit bottom
                reward -= 200.
            
            if crash or goal or obstacle or segment_frac_in_area < 1.0:
                break

        start_dist = self.distance(start_pos, [100., 80.])               # X=100,Y=80
        end_dist = self.distance(pos, [100., 80.])

        reward += start_dist - end_dist
        reward -= path_len
        
        
        return reward
    
    def get_dt(self):
        return self.dt_steps[min(int(self.time/10.), 4)]
    
    def step(self, action): # action can be enum of (1=rotate_left, -1=rotate_right, 0=dont_rotate)
        
        '''Calculate next position and velocity based on thrust, and action.
           Returns if the episode is done (due to timeout, crash, or goal reached)
        '''        
        
        self.dt = self.get_dt()
        
        # Gert the steering command from the passed action
        self.ang    = self.ang + self.dt/self.dt_ref*self.get_real_action(action)[0] + 0.01 * np.random.randn()   # consider actions
        self.thrust = self.get_real_action(action)[1]   
                    
        # Do the physics mechanics
        acc = self.thrust * self.get_trust_vector(self.ang)                
        self.vel += acc * self.dt                                                       
        segment = self.vel * self.dt 
        fraction = self.get_segment_fraction_in_area(self.pos, self.pos + segment)        
        self.pos  += segment * fraction
        self.time += self.dt * fraction

        # Check for events ;-)
        crash    = self.crashes(self.pos)
        goal     = self.reached_goal(self.pos)
        obstacle = self.obstacle(self.pos)
        
        # Get the (lookahead) reward
        reward = self.reward()
        
        if not self.has_hit_obstacle:
            self.has_hit_obstacle = obstacle
        
        # Check whether done
        done = (self.time > self.runtime) or crash or goal or obstacle or fraction < 1.0
        
        # Update remembered trajectory
        self.path.remember_step(self.pos, self.vel, acc, self.ang, self.time)
        
        # Remember the best trajectory
        if ((not self.has_hit_obstacle and goal) or self.demo) and done:
            if self.path.t < self.bestpath.t:
                self.bestpath = self.path        

        return self.get_state(), reward, done, None
        

    ###################################
    #  Draw helper and demo functions #
    ###################################
    
    def get_rotated_triangle(self, posx, posy, angle):
        
        ''' Helper function to traw a rotated triangle'''
        
        triangle_x = [-2., 0., 2., -2.]
        triangle_y = [-1., 2., -1., -1.]
        x_trans = []
        y_trans = []
        phi = (angle-90)/180*math.pi
        for x, y in zip(triangle_x, triangle_y):
            y_tmp =   y * math.cos(phi) + x * math.sin(phi)
            x_tmp =   x * math.cos(phi) - y * math.sin(phi)
            y_trans.append(y_tmp + posy)
            x_trans.append(x_tmp + posx)
        return x_trans, y_trans


    def plot_trajectory(self, title='Trajectory', trajectories=None, with_shuttle=True, num_episodes=None):
        
        ''' Helper function to plot trajectories with and without shuttle '''
        
        if trajectories is None:
            trajectories = []
            trajectories.append(self.bestpath)

        plt.figure(figsize=(8, 8))
        plt.axis('equal')
        
        colors=plt.cm.brg(np.linspace(0, 1, 1 if num_episodes is None else num_episodes))
        
        i = 0
        for trajectory in trajectories:
            posx, posy, velx, vely, accx, accy, angl, t = trajectory.gettrajectory()
            plt.plot(posx, posy, linewidth=1, alpha=0.5, color=colors[i])
            i += 1

        if with_shuttle:
            i = 0
            for ax, ay, px, py, ang in zip(accx, accy, posx, posy, angl):
                i += 1
                if i%(int(10*self.dt_ref/self.dt)) == 0:                          # do not draw the shuttle every time step, this would jam the picture
                    x = [px, px-80*ax]
                    y = [py, py-80*ay]
                    
                    plt.plot(x, y, color = 'red')                                 # draw the shuttles thrust
                    
                    shuttle_x, shuttle_y = self.get_rotated_triangle(px, py, ang) # draw the shuttle itself
                    plt.fill(shuttle_x, shuttle_y, color = 'blue')
                

        plt.plot(self.x_bound, self.y_bound, color = 'blue')

        plt.title(title)

        plt.show()


    def do_demo(self):
        
        ''' Provides a visual demo of the SpaceShuttlePhysics class by running an episode with the 
            shuttle's built in naive action policy.
            
            Draw the result
            
            The shuttle is denoted as a triangle
            The rocket thrust as a red vector at its back
            The trajectory is drawn as a light blue line 
            The worlds boundary as a solid blue line
        '''

        #
        # Simulate a whole episode ...
        #
        self.reset()
        self.demo = True
        self.angle = 90.

        while True:
            action = self.naive_policy_action() # get the naive action from the shuttles autopilot ;-)
            _,_,done,_ = self.step(action)      # let the simulation do the next action

            if done:
                break

        self.plot_trajectory('Shuttle Flight Demo (naive autopilot), t = {:.2f}'.format(self.time))
            
        plt.show()
    

        
