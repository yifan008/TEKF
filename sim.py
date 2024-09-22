#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
from tqdm import tqdm

from pathlib import Path

from ekf import Centralized_EKF
from fej import FEJ_EKF
from kdl import KD_LO_EKF
from inv import INV_EKF
from odom import ODOM_EKF
from ukf import UKF

from robot_system import *

import random

from matplotlib import pyplot as plt

def _set_axis(ax):
      for direction in ['left', 'right', 'top', 'bottom']:
          ax.spines[direction].set_linewidth(2.5)

      ax.tick_params(axis='both',
                      which='major',
                      direction='in',
                      length=6,
                      width=1.5,
                      colors='k')

      ax.tick_params(axis='both',
                      which='minor',
                      direction='in',
                      length=3,
                      width=1.5,
                      colors='k')

      ax.tick_params(labelsize=11) 

      ax.grid(color="gray", linestyle=':', linewidth=1)

class sim_mag:
    # robots team simulation
    class Robot():
        def __init__(self, robot_id):
            self.robot_id = robot_id
            
            self.p = np.random.uniform(-2, 2, (2, ))
            self.theta = np.random.uniform(-1, 1)

            self.v = np.array((0.3, 0.0))

            self.a = np.random.uniform(-0.5, 0.5)

            self.history = dict()

            self.history['p'] = list()
            self.history['theta'] = list()

            self.history['p'].append(np.copy(self.p))
            self.history['theta'].append(self.theta)

            self.history['v'] = list()
            self.history['a'] = list()

            self.history['v'].append(np.copy(self.v))
            self.history['a'].append(self.a)

    def __init__(self, team_settings):
        self.dt = team_settings['dt']
        self.duration = team_settings['duration']
        self.iter_num = team_settings['iter_num']

        self.robot_num = team_settings['robot_num']

        self.vx_sigma = team_settings['vx_sigma']
        self.vy_sigma = team_settings['vy_sigma']

        self.a_sigma = team_settings['a_sigma']

        self.pos_sigma = team_settings['p_sigma']

        # print(self.dt, self.duration, self.iter_num, self.robot_num, self.vx_sigma, self.vy_sigma, self.a_sigma)

        self.robots = list()

        for r in range(self.robot_num):
            robot = self.Robot(r+1)
            self.robots.append(robot)

    def motion_trajectory(self):
        # motion trajectories generation
        t = self.dt

        while t <= self.duration:
            for r in range(self.robot_num):
                self.robots[r].p += rot_mtx(self.robots[r].theta) @ self.robots[r].v * self.dt
                self.robots[r].theta += self.robots[r].a * self.dt
                
                # self.robots[r].v[0] += np.random.uniform(-0.01, 0.01) * self.dt
                # self.robots[r].v[1] += np.random.uniform(-0.01, 0.01) * self.dt
                
                self.robots[r].a = np.random.uniform(-0.5, 0.5) * self.dt

                self.robots[r].history['p'].append(np.copy(self.robots[r].p))
                self.robots[r].history['theta'].append(self.robots[r].theta)
                self.robots[r].history['v'].append(np.copy(self.robots[r].v))
                self.robots[r].history['a'].append(self.robots[r].a)
    
            t += self.dt

    def measurement_sim(self):
        # odometry information
        self.odometry = dict()

        for r in range(self.robot_num):
            ego_motion = dict()
            ego_motion['v'] = list()
            ego_motion['a'] = list()

            for k in range(len(self.robots[r].history['p'])):
                vx = self.robots[r].history['v'][k][0] + self.vx_sigma * np.random.randn()
                vy = self.robots[r].history['v'][k][1] + self.vy_sigma * np.random.randn()              
                
                az = self.robots[r].history['a'][k] + self.a_sigma * np.random.randn()
                
                ego_motion['v'].append(np.array((vx, vy)))

                ego_motion['a'].append(az)

            self.odometry[str(r+1)] = ego_motion

            # print(self.odometry[str(r+1)]['v'])

        # relative measurement  
        self.measurement = dict()
        self.measurement_prob = dict()

        for r in range(self.robot_num - 1): 

            self.measurement[str(r+1)] = list()
            self.measurement_prob[str(r+1)] = list()

            for k in range(len(self.robots[r].history['p'])):
                
                r_pos = dict()
                r_pos_prob = dict()

                r_pos['ij'] = rot_mtx(self.robots[r].history['theta'][k]).T @ (self.robots[r+1].history['p'][k] - self.robots[r].history['p'][k]) + np.random.normal(0, self.pos_sigma, (2, ))
                r_pos['ij'] += RANGE_DISTURB * np.random.uniform(-1.0, 1.0, (2, ))

                if np.random.uniform(0.0, 1.0) < prob:
                    r_pos_prob['ij'] = 1
                else:
                    r_pos_prob['ij'] = 0
               
                r_pos['ji'] = rot_mtx(self.robots[r+1].history['theta'][k]).T @ (self.robots[r].history['p'][k] - self.robots[r+1].history['p'][k]) + np.random.normal(0, self.pos_sigma, (2, ))
                r_pos['ji'] += RANGE_DISTURB * np.random.uniform(-1.0, 1.0, (2, ))

                if np.random.uniform(0.0, 1.0) < prob:
                    r_pos_prob['ji'] = 1
                else:
                    r_pos_prob['ji'] = 0

                self.measurement[str(r+1)].append(r_pos)
                self.measurement_prob[str(r+1)].append(r_pos_prob)
        
        return self.odometry, self.measurement, self.measurement_prob

if __name__ == '__main__':
    dt = STEP
    duration = DURATION
    iter_num = ITER_NUM
    robot_num = NUM_ROBOTS
    vx_sigma = VX_Sigma
    vy_sigma = VY_Sigma
    a_sigma = VA_sigma
    p_sigma = P_Sigma

    # sim settings
    sim_settings = {'dt': dt, 'duration': duration, 'iter_num': iter_num, 'robot_num': robot_num, 'vx_sigma': vx_sigma, 'vy_sigma': vy_sigma, 'a_sigma': a_sigma, 'p_sigma': p_sigma}

    sim = sim_mag(sim_settings)

    sim.motion_trajectory()

    xyt_0 = np.zeros((3*robot_num))

    gt = dict()

    for r in range(robot_num):
        xyt_0[3*r:3*r+2] = sim.robots[r].history['p'][0]
        xyt_0[3*r+2] = sim.robots[r].history['theta'][0]

        gt[str(r+1)] = dict()

        gt[str(r+1)]['p'] = sim.robots[r].history['p']
        gt[str(r+1)]['theta'] = sim.robots[r].history['theta']

    # algorithms = ['ekf', 'inv', 'kdl', 'ukf', 'iukf', 'tukf'] #  'fej' 'odom', 
    
    algorithms = ['odom', 'ekf', 'fej', 'inv', 'kdl', 'ukf']

    # algorithms = ['odom', 'ekf', 'fej', 'inv', 'kd', 'fej3', 'kd3']

    # algorithms = ['ekf', 'fej', 'kdg', 'kdl', 'kd', 'inv']

    # algorithms = ['ekf', 'fej', 'inv', 'kd', 'kdp', 'kdg', 'kdl']

    team_settings = {'dt': dt, 'duration': duration, 'iter_num': iter_num, 'robot_num': robot_num, 'vx_sigma': vx_sigma, 'vy_sigma': vy_sigma, 'a_sigma': a_sigma, 'p_sigma': p_sigma}

    results = list()

    with tqdm(total=(iter_num), leave=False) as pbar:
        for i in range(iter_num):
            odometry, measurement, measurement_prob = sim.measurement_sim()

            print('algorithm running ...')

            dataset = dict()
            dataset['odometry'] = odometry
            dataset['measurement'] = measurement
            dataset['measurement_prob'] = measurement_prob
            dataset['gt'] = gt

            result_alg = dict()

            for alg in algorithms:
                if alg == 'ekf':
                    robot_system = CentralizedSystem(xyt_0, team_settings)
                    CENEKF = Centralized_EKF(robot_system, dataset)
                    CENEKF.run()
                    robot_system = CENEKF.robot_system
                elif alg == 'odom':
                    robot_system = CentralizedSystem(xyt_0, team_settings)
                    ODEKF = ODOM_EKF(robot_system, dataset)
                    ODEKF.run()
                    robot_system = ODEKF.robot_system
                elif alg == 'fej':
                    robot_system = CentralizedSystem(xyt_0, team_settings)
                    FEJEKF = FEJ_EKF(robot_system, dataset)
                    FEJEKF.run()
                    robot_system = FEJEKF.robot_system
                elif alg == 'kdl':
                    robot_system = CentralizedSystem(xyt_0, team_settings)
                    KDLEKF = KD_LO_EKF(robot_system, dataset)
                    KDLEKF.run()
                    robot_system = KDLEKF.robot_system
                elif alg == 'inv':
                    robot_system = CentralizedSystem(xyt_0, team_settings)
                    INVEKF = INV_EKF(robot_system, dataset)
                    INVEKF.run()
                    robot_system = INVEKF.robot_system
                elif alg == 'ukf':
                    robot_system = CentralizedSystem(xyt_0, team_settings)
                    UKF_ = UKF(robot_system, dataset)
                    UKF_.run()
                    robot_system = UKF_.robot_system
                else:
                    sys.exit('Invalid algorithm input!')
            
                result_alg[alg] = robot_system

            results.append(result_alg)

            pbar.update()

    # individual RMSE plots
    rmse_pos = dict()
    rmse_ori = dict()
    nees_avg_pos = dict()
    nees_avg_ori = dict()
        
    for alg in algorithms:
        rmse_pos[alg] = 0
        rmse_ori[alg] = 0
        nees_avg_pos[alg] = 0
        nees_avg_ori[alg] = 0

    t = time.strftime("%Y-%m-%d %H:%M:%S")

    print('TIME: {}'.format(t))

    for r in range(robot_num):
        x_gt = np.squeeze([sim.robots[r].history['p'][k][0] for k in range(len(sim.robots[r].history['p']))])
        y_gt = np.squeeze([sim.robots[r].history['p'][k][1] for k in range(len(sim.robots[r].history['p']))])
        theta_gt = np.squeeze([sim.robots[r].history['theta'][k] for k in range(len(sim.robots[r].history['theta']))])

        time_arr = np.array([k * dt for k in range(int(duration / dt + 1))])

        for alg in algorithms:

            pos_error = np.zeros((int(duration / dt + 1), ))
            ori_error = np.zeros((int(duration / dt + 1), ))
          
            nees_pos = np.zeros((int(duration / dt + 1), ))
            nees_ori = np.zeros((int(duration / dt + 1), ))

            for i in range(iter_num):
                s_nees = np.zeros((int(duration / dt + 1), ))
                s_nees_ori = np.zeros((int(duration / dt + 1), ))
                s_nees_pos = np.zeros((int(duration / dt + 1), ))

                time_arr = np.array([k * dt for k in range(int(duration / dt + 1))])

                x_est = np.squeeze([results[i][alg].robots[r].history[k]['x'] for k in range(len(results[i][alg].robots[r].history))])
                y_est = np.squeeze([results[i][alg].robots[r].history[k]['y'] for k in range(len(results[i][alg].robots[r].history))])
                theta_est = np.squeeze([results[i][alg].robots[r].history[k]['theta'] for k in range(len(results[i][alg].robots[r].history))])
                
                cov_x_est = np.squeeze([results[i][alg].robots[r].history[k]['cov'][0, 0] for k in range(len(results[i][alg].robots[r].history))])
                cov_y_est = np.squeeze([results[i][alg].robots[r].history[k]['cov'][1, 1] for k in range(len(results[i][alg].robots[r].history))])
                cov_theta_est = np.squeeze([results[i][alg].robots[r].history[k]['cov'][2, 2] for k in range(len(results[i][alg].robots[r].history))])

                epos = (np.array(x_est) - np.array(x_gt)) ** 2 + (np.array(y_est) - np.array(y_gt)) ** 2
                pos_error += epos
                eori = (np.arctan2(np.sin(theta_est - theta_gt), np.cos(theta_est - theta_gt))) ** 2
                ori_error += eori

                for k in range(len(results[i][alg].robots[r].history)):
                    cov = results[i][alg].robots[r].history[k]['cov']

                    dx = np.array((x_est[k] - x_gt[k], y_est[k] - y_gt[k]))
                    d_theta = m.atan2(m.sin(theta_est[k] - theta_gt[k]), m.cos(theta_est[k] - theta_gt[k]))

                    nees_pos[k] += dx.T @ np.linalg.inv(cov[0:2, 0:2]) @ dx
                    nees_ori[k] += d_theta **2 / cov[2, 2]

                    ds = np.array((x_est[k] - x_gt[k], y_est[k] - y_gt[k], d_theta))
                    s_nees[k] = ds.T @ np.linalg.inv(cov) @ ds
                    s_nees_pos[k] = dx.T @ np.linalg.inv(cov[0:2, 0:2]) @ dx
                    s_nees_ori[k] = d_theta **2 / cov[2, 2]
                    
                save_path = '../sim_results' + '/' + t + '/MRCLAM' + str(i+1) + '/' 

                Path(save_path).mkdir(parents=True, exist_ok=True)

                file_name = alg + "_robot" + str(r+1) + '.npz'

                np.savez(save_path + file_name, t = time_arr, x_est = x_est, y_est = y_est, theta_est = theta_est,
                         x_gt = x_gt, y_gt = y_gt, theta_gt = theta_gt,
                         cov_x_est = cov_x_est, cov_y_est = cov_y_est, cov_theta_est = cov_theta_est, 
                         epos = epos, eori = eori, nees = s_nees, nees_ori = s_nees_ori, nees_pos = s_nees_pos)

            rmse_pos[alg] += np.sum(pos_error)
            rmse_ori[alg] += np.sum(ori_error)

            nees_avg_pos[alg] += np.sum(nees_pos)
            nees_avg_ori[alg] += np.sum(nees_ori)
            
            N = time_arr.shape[0]

    data_num = robot_num * iter_num * N

    print('VAR: {} {} {}'.format(SENSOR_VAR_X, ORIENTATION_VAR, POS_VAR))

    print('ALG: RMSE_POS             RMSE_ORI             NEES_POS               NEES_ORI')
    
    # num = int(RANGE_DISTURB / 0.5 + 1)
    num = int(round(prob / 0.2))

    for alg in algorithms:

        print('data[\'{}_{}\'] = np.array([{}, {}, {}, {}])'.format(alg, num, np.sqrt(rmse_pos[alg] / data_num), np.sqrt(rmse_ori[alg] / data_num), nees_avg_pos[alg] / data_num, nees_avg_ori[alg] / data_num))
