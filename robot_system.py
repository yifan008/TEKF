#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math as m

from params import *

def rot_mtx(theta):
    mtx = np.array([[m.cos(theta), -m.sin(theta)], [m.sin(theta), m.cos(theta)]])

    return mtx

class RobotSystem():

    def __init__(self, dataset, team_settings):
        self.robots = self.init_robots()
        
        self.dt = team_settings['dt']
        self.duration = team_settings['duration']

    def init_robots(self):
        robots = list()

        for r in range(NUM_ROBOTS):
            robot = self.Robot(r+1)
            robots.append(robot)

        return robots

class CentralizedSystem(RobotSystem):

    class Robot():
        def __init__(self, robot_id):
            self.robot_id = robot_id

            self.history = list()

    def __init__(self, dataset, team_settings):
        RobotSystem.__init__(self, dataset, team_settings)

        self.xyt = np.zeros(shape=(3*NUM_ROBOTS), dtype=float)
        self.cov =  np.zeros(shape=(3*NUM_ROBOTS, 3*NUM_ROBOTS), dtype=float)

        self.init_team(dataset)

    def init_team(self, dataset):

        for r in range(NUM_ROBOTS):
            x = dataset[3*r]
            y = dataset[3*r+1]
            theta = dataset[3*r+2]

            self.xyt[3*r:3*(r+1)] = np.array((x, y, theta))

            self.robots[r].history.append({'x': x, 'y': y, 'theta': theta, 'cov': np.identity(3) * 1e-6})
        
        # print(self.xyt)

        self.cov = np.identity(3*NUM_ROBOTS) * 1e-6