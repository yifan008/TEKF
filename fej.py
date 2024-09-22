#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import math as m
import time

from tqdm import tqdm

from robot_system import *

class FEJ_EKF():
    def __init__(self, robot_system, dataset):
        self.robot_system = robot_system
        self.dataset = dataset

        self.dt = self.robot_system.dt
        self.duration = self.robot_system.duration
        
        self.Q = np.identity(3*NUM_ROBOTS)
        self.F = np.identity(3*NUM_ROBOTS)

        self.J = np.array([[0, -1], [1, 0]])

        self.xyt_last = np.zeros(shape=(3*NUM_ROBOTS), dtype=float)
        
        self.prediction_flag = False

    def prediction(self, t):
        dataset = self.dataset

        for r in range(NUM_ROBOTS):
          # get nearest index
          idx = int(t / self.dt - 1)

          # extract odometry
          v = dataset['odometry'][str(r+1)]['v'][idx]
          a = dataset['odometry'][str(r+1)]['a'][idx]

          # extract state vector
          x = self.robot_system.xyt[3*r]
          y = self.robot_system.xyt[3*r+1]
          
          p = np.array((x, y))

          theta = self.robot_system.xyt[3*r+2]

          # noise covariance matrix
          G = rot_mtx(theta)

          W = np.identity(2) 
          W[0, 0] = SENSOR_VAR_X * self.dt**2
          W[1, 1] = SENSOR_VAR_Y * self.dt**2

          self.Q[3*r:3*r+2, 3*r:3*r+2] = G @ W @ G.T
          self.Q[3*r+2, 3*r+2] = ORIENTATION_VAR * self.dt**2

          # jacobian matrix
          if not self.prediction_flag:
            self.F[3*r:3*r+2, 3*r+2] = self.J @ rot_mtx(theta) @ v * self.dt

            # state prediction
            p += rot_mtx(theta) @ v * self.dt
            theta += a * self.dt

            self.robot_system.xyt[3*r:3*r+3] = np.array((p[0], p[1], theta))

            self.xyt_last[3*r:3*r+3] = np.array((p[0], p[1], theta))
          else:
            # state prediction
            p += rot_mtx(theta) @ v * self.dt
            theta += a * self.dt

            self.robot_system.xyt[3*r:3*(r+1)] = np.array((p[0], p[1], theta))

            # jacobian matrix
            self.F[3*r:3*r+2, 3*r+2] = self.J @ (p - self.xyt_last[3*r:3*r+2])

            self.xyt_last[3*r:3*r+3] = np.array((p[0], p[1], theta))

        # covariance prediction
        self.robot_system.cov = self.F @ self.robot_system.cov @ self.F.T + self.Q

        # prediction flag ---> intialize
        if not self.prediction_flag:
          self.prediction_flag = True
    
    def relative_observation(self, t):
        dataset = self.dataset

        if MType == 'pos':
            # print('pos')
            H = np.zeros((2*2*(NUM_ROBOTS - 1), 3*NUM_ROBOTS))

            dz = np.zeros((2*2*(NUM_ROBOTS - 1))) 

            R = np.identity(2*2*(NUM_ROBOTS - 1))

            for r in range(NUM_ROBOTS-1):
                # get nearest index
                idx = int(t / self.dt)

                # extract robot measurements
                r_pos_ij = dataset['measurement'][str(r+1)][idx]['ij']
                r_pos_ji = dataset['measurement'][str(r+1)][idx]['ji']

                r_pos_ij_prob = dataset['measurement_prob'][str(r+1)][idx]['ij']
                r_pos_ji_prob = dataset['measurement_prob'][str(r+1)][idx]['ji']

                # robot location
                x = self.robot_system.xyt[3*r]
                y = self.robot_system.xyt[3*r+1]
                theta = self.robot_system.xyt[3*r+2]

                # observed robot location
                r_j = r+1
                x_j = self.robot_system.xyt[3*r_j]
                y_j = self.robot_system.xyt[3*r_j+1]
                theta_j = self.robot_system.xyt[3*r_j+2]

                # predicted measurement
                dx = x_j - x
                dy = y_j - y
                
                z_hat_ij = rot_mtx(theta).T @ np.array((dx, dy))
                z_hat_ji = rot_mtx(theta_j).T @ np.array((-dx, -dy))

                dz_ij = r_pos_ij - z_hat_ij
                dz_ji = r_pos_ji - z_hat_ji

                # construct measurement matrix
                # H_ij
                H_ij = np.zeros((2, 3*NUM_ROBOTS))

                H_ij[0:2, 3*r:3*r+2] = - np.identity(2)
                H_ij[0:2, 3*r+2] = - self.J @ np.array((dx, dy))
                
                H_ij[0:2, 3*r_j:3*r_j+2] = np.identity(2)

                H_ij = rot_mtx(theta).T @ H_ij

                R_ij = POS_VAR * np.identity(2)

                # H_ji
                H_ji = np.zeros((2, 3*NUM_ROBOTS))

                H_ji[0:2, 3*r_j:3*r_j+2] = - np.identity(2)
                H_ji[0:2, 3*r_j+2] = - self.J @ np.array((-dx, -dy))
                
                H_ji[0:2, 3*r:3*r+2] = np.identity(2)

                H_ji = rot_mtx(theta_j).T @ H_ji

                R_ji = POS_VAR * np.identity(2)

                if r_pos_ij_prob == 0:
                    H_ij = 0 * H_ij

                if r_pos_ji_prob == 0:
                    H_ji = 0 * H_ji

                dz[4*r:4*r+2] = dz_ij[:]
                dz[4*r+2:4*r+4] = dz_ji[:]

                H[4*r:4*r+2, :] = H_ij[0:2, :]
                H[4*r+2:4*r+4, :] = H_ji[0:2, :]

                R[4*r:4*r+2, 4*r:4*r+2] = R_ij
                R[4*r+2:4*r+4, 4*r+2:4*r+4] = R_ji

            cov = self.robot_system.cov

            innovation = H @ cov @ H.T + R

            Kalman_gain = cov @ H.T @ np.linalg.inv(innovation)

            self.robot_system.xyt += Kalman_gain @ dz
            self.robot_system.cov = cov - Kalman_gain @ H @ cov

        else:
            exit('MType class error')

    def save_est(self, t):
        for r in range(NUM_ROBOTS):
          # save into history
          x = self.robot_system.xyt[3*r]
          y = self.robot_system.xyt[3*r+1]
          theta = self.robot_system.xyt[3*r+2]

          cov = self.robot_system.cov[3*r:3*r+3, 3*r:3*r+3]

          self.robot_system.robots[r].history.append({'x': np.copy(x), 'y': np.copy(y), 'theta': np.copy(theta), 'cov': np.copy(cov)})

    def run(self):
        # initialize time
        t = self.dt
        
        start_time = time.time()

        while t <= self.duration: 

          # prediction (time propagation) step
          self.prediction(t)

          # relative observation update (other robots)
          self.relative_observation(t)

          # save the estimate
          self.save_est(t)

          # update the time
          t = t + self.dt
        
        end_time = time.time()
        # print('fej duration: {} \n'.format((end_time - start_time) / self.duration * self.dt))
