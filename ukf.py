#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

from robot_system import *

class UKF():
    def __init__(self, robot_system, dataset):
        self.robot_system = robot_system
        self.dataset = dataset

        self.dt = self.robot_system.dt
        self.duration = self.robot_system.duration
        
        self.Q = np.identity(3*NUM_ROBOTS)
        self.F = np.identity(3*NUM_ROBOTS)

        self.J = np.array([[0, -1], [1, 0]])

        self.kappa = 0.5

    def prediction_(self, t):
        dataset = self.dataset

        dim = 3*NUM_ROBOTS + 3*NUM_ROBOTS
        u = np.zeros((dim, ))
        sigma = np.identity(dim)
        sigma[0:3*NUM_ROBOTS, 0:3*NUM_ROBOTS] = self.robot_system.cov
        u[0:3*NUM_ROBOTS] = self.robot_system.xyt

        for r in range(NUM_ROBOTS):
          sigma[3*NUM_ROBOTS + 3*r, 3*NUM_ROBOTS + 3*r] = SENSOR_VAR_X
          sigma[3*NUM_ROBOTS + 3*r+1, 3*NUM_ROBOTS + 3*r+1] = SENSOR_VAR_Y
          sigma[3*NUM_ROBOTS + 3*r+2, 3*NUM_ROBOTS + 3*r+2] = ORIENTATION_VAR

        L = np.linalg.cholesky(sigma)

        u_p = list()
        u_l = list()

        for l in range(dim):
          sigma_p = u + np.sqrt(dim + self.kappa) * L[:, l]
          sigma_l = u - np.sqrt(dim + self.kappa) * L[:, l]

          u_p.append(sigma_p)
          u_l.append(sigma_l)

        v = np.zeros((2*NUM_ROBOTS, ))
        a = np.zeros((NUM_ROBOTS, ))

        for r in range(NUM_ROBOTS):
            # get nearest index
            idx = int(t / self.dt - 1)
            # extract odometry
            v[2*r:2*(r+1)] = dataset['odometry'][str(r+1)]['v'][idx]
            a[r] = dataset['odometry'][str(r+1)]['a'][idx]

        s = np.zeros((3*NUM_ROBOTS, ))
        s_p = list()
        s_l = list()

        for l in range(dim):
          x_p = np.zeros((3*NUM_ROBOTS, ))
          x_l = np.zeros((3*NUM_ROBOTS, ))
            
          for r in range(NUM_ROBOTS):  
            x_p[3*r:3*r+2] = u_p[l][3*r:3*r+2] + rot_mtx(u_p[l][3*r+2]) @ (v[2*r:2*(r+1)] + u_p[l][(3*NUM_ROBOTS + 3*r):(3*NUM_ROBOTS + 3*r+2)]) * self.dt
            x_p[3*r+2] = u_p[l][3*r+2] + (a[r] + u_p[l][3*NUM_ROBOTS + 3*r+2]) * self.dt

            x_l[3*r:3*r+2] = u_l[l][3*r:3*r+2] + rot_mtx(u_l[l][3*r+2]) @ (v[2*r:2*(r+1)] + u_l[l][(3*NUM_ROBOTS + 3*r):(3*NUM_ROBOTS + 3*r+2)]) * self.dt
            x_l[3*r+2] = u_l[l][3*r+2] + (a[r] + u_l[l][3*NUM_ROBOTS + 3*r+2]) * self.dt

          s_p.append(x_p)
          s_l.append(x_l)

        for r in range(NUM_ROBOTS):  
          s[3*r:3*r+2] = u[3*r:3*r+2] + rot_mtx(u[3*r+2]) @ (v[2*r:2*(r+1)] + u[(3*NUM_ROBOTS + 3*r):(3*NUM_ROBOTS + 3*r+2)]) * self.dt
          s[3*r+2] = u[3*r+2] + (a[r] + u[3*NUM_ROBOTS + 3*r+2]) * self.dt

        s_average = s * self.kappa / (self.kappa + dim)

        for l in range(dim):
          s_average += s_p[l] /  (self.kappa + dim) / 2
          s_average += s_l[l] /  (self.kappa + dim) / 2

        e = s - s_average
        ez = np.zeros((3*NUM_ROBOTS, 1))
        ez[:, 0] = e

        cov = ez @ ez.T * self.kappa / (self.kappa + dim)

        for l in range(dim):
          e = s_p[l] - s_average
          ez = np.zeros((3*NUM_ROBOTS, 1))
          ez[:, 0] = e
       
          cov += ez @ ez.T /  (self.kappa + dim) / 2

          e = s_l[l] - s_average
          ez = np.zeros((3*NUM_ROBOTS, 1))
          ez[:, 0] = e
          
          cov += ez @ ez.T /  (self.kappa + dim) / 2

        self.robot_system.xyt = np.identity(3*NUM_ROBOTS) @ s_average
        self.robot_system.cov = np.identity(3*NUM_ROBOTS) @ cov


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

            # jacobian matrix
            self.F[3*r:3*r+2, 3*r+2] = self.J @ rot_mtx(theta) @ v * self.dt

            # noise covariance matrix
            G = rot_mtx(theta)
            
            W = np.identity(2) 
            W[0, 0] = SENSOR_VAR_X * self.dt**2
            W[1, 1] = SENSOR_VAR_Y * self.dt**2

            self.Q[3*r:3*r+2, 3*r:3*r+2] = G @ W @ G.T
            self.Q[3*r+2, 3*r+2] = ORIENTATION_VAR * self.dt**2

            # state prediction
            p += rot_mtx(theta) @ v * self.dt
            theta += a * self.dt

            self.robot_system.xyt[3*r:3*(r+1)] = np.array((p[0], p[1], theta))

        # covariance prediction
        self.robot_system.cov = self.F @ self.robot_system.cov @ self.F.T + self.Q

    def relative_observation(self, t):
        dataset = self.dataset

        if MType == 'pos':
            # print('pos')

            m_dim = 0

            for r in range(NUM_ROBOTS-1):
                # get nearest index
                idx = int(t / self.dt)

                r_pos_ij_prob = dataset['measurement_prob'][str(r+1)][idx]['ij']
                r_pos_ji_prob = dataset['measurement_prob'][str(r+1)][idx]['ji']

                if r_pos_ij_prob != 0:
                    m_dim += 1

                if r_pos_ji_prob != 0:
                    m_dim += 1

            if m_dim == 0:
              return

            y = np.zeros((2*m_dim, ))

            dim_flag = 0

            for r in range(NUM_ROBOTS-1):
                # get nearest index
                idx = int(t / self.dt)

                # extract robot measurements
                r_pos_ij = dataset['measurement'][str(r+1)][idx]['ij']
                r_pos_ji = dataset['measurement'][str(r+1)][idx]['ji']

                r_pos_ij_prob = dataset['measurement_prob'][str(r+1)][idx]['ij']
                r_pos_ji_prob = dataset['measurement_prob'][str(r+1)][idx]['ji']

                if r_pos_ij_prob != 0:
                    y[2*dim_flag:2*(dim_flag+1)] = r_pos_ij
                    dim_flag += 1

                if r_pos_ji_prob != 0:
                    y[2*dim_flag:2*(dim_flag+1)] = r_pos_ji
                    dim_flag += 1

            dim = 3*NUM_ROBOTS + 2*m_dim
            u = np.zeros((dim, ))
            sigma = np.identity(dim)
            sigma[0:3*NUM_ROBOTS, 0:3*NUM_ROBOTS] = self.robot_system.cov
            sigma[3*NUM_ROBOTS:dim, 3*NUM_ROBOTS:dim] = POS_VAR * np.identity(dim - 3*NUM_ROBOTS)
            u[0:3*NUM_ROBOTS] = self.robot_system.xyt

            L = np.linalg.cholesky(sigma)

            u_p = list()
            u_l = list()

            for l in range(dim):
              sigma_p = u + np.sqrt(dim + self.kappa) * L[:, l]
              sigma_l = u - np.sqrt(dim + self.kappa) * L[:, l]

              u_p.append(sigma_p)
              u_l.append(sigma_l)

            y_ = np.zeros((2*m_dim, ))
            ly_p = list()
            ly_l = list()

            for l in range(dim):
              y_p = np.zeros((2*m_dim, ))
              y_l = np.zeros((2*m_dim, ))
                
              dim_flag = 0

              for r in range(NUM_ROBOTS-1):
                # get nearest index
                idx = int(t / self.dt)

                r_pos_ij_prob = dataset['measurement_prob'][str(r+1)][idx]['ij']
                r_pos_ji_prob = dataset['measurement_prob'][str(r+1)][idx]['ji']

                if r_pos_ij_prob != 0:
                    r_j = r+1
                    
                    x_i = u_p[l][3*r]
                    y_i = u_p[l][3*r+1]
                    theta_i = u_p[l][3*r+2]

                    x_j = u_p[l][3*r_j]
                    y_j = u_p[l][3*r_j+1]
                    theta_j = u_p[l][3*r_j+2]

                    dx = x_j - x_i
                    dy = y_j - y_i
                    
                    y_p[2*dim_flag:2*(dim_flag+1)] = rot_mtx(theta_i).T @ np.array((dx, dy)) + u_p[l][(3*NUM_ROBOTS + 2*dim_flag):(3*NUM_ROBOTS + 2*(dim_flag+1))]
                    
                    x_i = u_l[l][3*r]
                    y_i = u_l[l][3*r+1]
                    theta_i = u_l[l][3*r+2]

                    x_j = u_l[l][3*r_j]
                    y_j = u_l[l][3*r_j+1]
                    theta_j = u_l[l][3*r_j+2]

                    dx = x_j - x_i
                    dy = y_j - y_i
                    
                    y_l[2*dim_flag:2*(dim_flag+1)] = rot_mtx(theta_i).T @ np.array((dx, dy)) + u_l[l][(3*NUM_ROBOTS + 2*dim_flag):(3*NUM_ROBOTS + 2*(dim_flag+1))]
                    
                    dim_flag += 1

                if r_pos_ji_prob != 0:
                    r_j = r+1
                    
                    x_i = u_p[l][3*r]
                    y_i = u_p[l][3*r+1]
                    theta_i = u_p[l][3*r+2]

                    x_j = u_p[l][3*r_j]
                    y_j = u_p[l][3*r_j+1]
                    theta_j = u_p[l][3*r_j+2]

                    dx = x_j - x_i
                    dy = y_j - y_i
                    
                    y_p[2*dim_flag:2*(dim_flag+1)] = rot_mtx(theta_j).T @ np.array((-dx, -dy)) + u_p[l][(3*NUM_ROBOTS + 2*dim_flag):(3*NUM_ROBOTS + 2*(dim_flag+1))]

                    x_i = u_l[l][3*r]
                    y_i = u_l[l][3*r+1]
                    theta_i = u_l[l][3*r+2]

                    x_j = u_l[l][3*r_j]
                    y_j = u_l[l][3*r_j+1]
                    theta_j = u_l[l][3*r_j+2]

                    dx = x_j - x_i
                    dy = y_j - y_i

                    y_l[2*dim_flag:2*(dim_flag+1)] = rot_mtx(theta_j).T @ np.array((-dx, -dy)) + u_l[l][(3*NUM_ROBOTS + 2*dim_flag):(3*NUM_ROBOTS + 2*(dim_flag+1))]

                    dim_flag += 1

              ly_p.append(y_p)
              ly_l.append(y_l)

            dim_flag = 0

            for r in range(NUM_ROBOTS-1):
              # get nearest index
              idx = int(t / self.dt)

              r_pos_ij_prob = dataset['measurement_prob'][str(r+1)][idx]['ij']
              r_pos_ji_prob = dataset['measurement_prob'][str(r+1)][idx]['ji']

              if r_pos_ij_prob != 0:
                  r_j = r+1
                  
                  x_i = u[3*r]
                  y_i = u[3*r+1]
                  theta_i = u[3*r+2]

                  x_j = u[3*r_j]
                  y_j = u[3*r_j+1]
                  theta_j = u[3*r_j+2]

                  dx = x_j - x_i
                  dy = y_j - y_i
                  
                  y_[2*dim_flag:2*(dim_flag+1)] = rot_mtx(theta_i).T @ np.array((dx, dy)) + u[(3*NUM_ROBOTS + 2*dim_flag):(3*NUM_ROBOTS + 2*(dim_flag+1))]
                  
                  dim_flag += 1

              if r_pos_ji_prob != 0:
                  r_j = r+1
                  
                  x_i = u[3*r]
                  y_i = u[3*r+1]
                  theta_i = u[3*r+2]

                  x_j = u[3*r_j]
                  y_j = u[3*r_j+1]
                  theta_j = u[3*r_j+2]

                  dx = x_j - x_i
                  dy = y_j - y_i
                  
                  y_[2*dim_flag:2*(dim_flag+1)] = rot_mtx(theta_j).T @ np.array((-dx, -dy)) + u[(3*NUM_ROBOTS + 2*dim_flag):(3*NUM_ROBOTS + 2*(dim_flag+1))]

                  # print(y_)

                  dim_flag += 1

            y_average = y_ * self.kappa / (self.kappa + dim)

            
            for l in range(dim):
              # print(y_average.shape, ly_p[l].shape)
              y_average += ly_p[l] / (self.kappa + dim) / 2
              y_average += ly_l[l] / (self.kappa + dim) / 2

            e = y_ - y_average
            ez = np.zeros((2*m_dim, 1))
            ez[:, 0] = e

            cov_yy = ez @ ez.T * self.kappa / (self.kappa + dim)

            for l in range(dim):
              e = ly_p[l] - y_average
              ez = np.zeros((2*m_dim, 1))
              ez[:, 0] = e
          
              cov_yy += ez @ ez.T /  (self.kappa + dim) / 2

              e = ly_l[l] - y_average
              ez = np.zeros((2*m_dim, 1))
              ez[:, 0] = e
              
              cov_yy += ez @ ez.T /  (self.kappa + dim) / 2

            ey_ = y_ - y_average
            eu_ = u[0:3*NUM_ROBOTS] - u[0:3*NUM_ROBOTS]
            ey = np.zeros((2*m_dim, 1))
            ey[:, 0] = ey_
            eu = np.zeros((3*NUM_ROBOTS, 1))
            eu[:, 0] = eu_

            cov_uy = eu @ ey.T * self.kappa / (self.kappa + dim)

            for l in range(dim):
          
              ey_ = ly_p[l] - y_average
              eu_ = u_p[l][0:3*NUM_ROBOTS] - u[0:3*NUM_ROBOTS]
              ey = np.zeros((2*m_dim, 1))
              ey[:, 0] = ey_
              eu = np.zeros((3*NUM_ROBOTS, 1))
              eu[:, 0] = eu_

              cov_uy += eu @ ey.T /  (self.kappa + dim) / 2
              
              ey_ = ly_l[l] - y_average
              eu_ = u_l[l][0:3*NUM_ROBOTS] - u[0:3*NUM_ROBOTS]
              ey = np.zeros((2*m_dim, 1))
              ey[:, 0] = ey_
              eu = np.zeros((3*NUM_ROBOTS, 1))
              eu[:, 0] = eu_

              cov_uy += eu @ ey.T /  (self.kappa + dim) / 2


            cov_xx = self.robot_system.cov

            Kalman_gain = cov_uy @ np.linalg.inv(cov_yy)

            self.robot_system.xyt = u[0:3*NUM_ROBOTS] + Kalman_gain @ (y - y_average)
            self.robot_system.cov = cov_xx - Kalman_gain @ cov_uy.T
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
 
        # print('ukf duration: {} \n'.format((end_time - start_time) / self.duration * self.dt))
