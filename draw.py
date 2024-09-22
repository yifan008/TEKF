#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
from tqdm import tqdm

from pathlib import Path

from ideal import Ideal_EKF
from ekf import Centralized_EKF
from fej import FEJ_EKF
from kdg import KD_GO_EKF
from kdl import KD_LO_EKF
from kdp import KD_GP_EKF
from kd import KD_EKF
from inv import INV_EKF

from robot_system import *

import random


from matplotlib import markers, pyplot as plt

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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

      # ax.grid(color="gray", linestyle=':', linewidth=1)

if __name__ == '__main__':
    # t = time.strftime("%Y-%m-%d %H:%M:%S")
    t = TIME_MARK

    # algorithms = ['ekf', 'fej', 'inv', 'kd', 'kdg', 'kdg2', 'kdl', 'kdl2']
    algorithms = ['ukf', 'ekf', 'fej', 'inv', 'kdl']
    
    # algorithms = ['ekf', 'fej', 'inv', 'kd', 'kdg', 'kdp', 'kdl']
    color_tables = {'ekf':'blue', 'fej':'lime', 'inv':'orange', 'kd':'purple', 'kdl':'red', 'kdg':'hotpink', 'kdp':'hotpink', 'odom':'yellow', 'gt':'red', 'kdl2':'Moccasin', 'kdg2':'LavenderBlush', 'ukf':'Magenta'}
    marker_tables = {'ekf':'o', 'fej':'h', 'inv':'s', 'kd':'^', 'kdl':'p', 'kdg':'3', 'kdp':'*', 'gt':'2', 'odom':'o', 'kdg2':'s', 'ukf':'*'}
    label_tables = {'ekf':'EKF', 'fej':'FEJ', 'inv':'I-EKF', 'kd':'KD', 'kdl':'T-EKF', 'ukf':'UKF', 'kdp':'T-EKF', 'odom':'ODOM'} 
    style_table = {'ekf':'-', 'fej':'--', 'inv':'-.', 'ukf':':', 'kdl':':', 'kd':':'}

    robot_num = NUM_ROBOTS
    iter_num = ITER_NUM

    # individual RMSE plots
    rmse_pos = dict()
    rmse_ori = dict()
    nees_avg = dict()
    nees_pos = dict()
    nees_ori = dict()

    rpe_pos = dict()
    rpe_ori = dict()

    for i in range(iter_num):
      for alg in algorithms:
        for r in range(robot_num):
          save_path = '../sim_results' + '/' + t + '/MRCLAM' + str(i+1) + '/'
          file_name = alg + "_robot" + str(r+1) + '.npz'

          data = np.load(save_path + file_name)

          time_arr = data['t']

          pos_error = data['epos']
          ori_error = data['eori']
          s_nees = data['nees']
          s_nees_pos = data['nees_pos']
          s_nees_ori = data['nees_ori']

          x_gt = data['x_gt']
          y_gt = data['y_gt']
          theta_gt = data['theta_gt']

          x_est = data['x_est']
          y_est = data['y_est']
          theta_est = data['theta_est']

          N = time_arr.shape[0]

          if alg not in rmse_pos and alg not in rmse_ori:
            rmse_pos[alg] = pos_error
            rmse_ori[alg] = ori_error
            nees_avg[alg] = s_nees
            nees_pos[alg] = s_nees_pos
            nees_ori[alg] = s_nees_ori

            rpe_pos[alg] = ((x_est[1:N] - x_est[0:N-1]) - (x_gt[1:N] - x_gt[0:N-1])) **2 + ((y_est[1:N] - y_est[0:N-1]) - (y_gt[1:N] - y_gt[0:N-1])) **2
            # rpe_ori[alg] = ((theta_est[1:N] - theta_est[0:N-1]) - (theta_gt[1:N] - theta_gt[0:N-1])) **2
            rpe_theta = (theta_est[1:N] - theta_est[0:N-1]) - (theta_gt[1:N] - theta_gt[0:N-1])
            rpe_ori[alg] = np.arctan2(np.sin(rpe_theta), np.cos(rpe_theta)) **2 
          else:
            rmse_pos[alg] += pos_error
            rmse_ori[alg] += ori_error
            nees_avg[alg] += s_nees
            nees_pos[alg] += s_nees_pos
            nees_ori[alg] += s_nees_ori

            rpe_pos[alg] += ((x_est[1:N] - x_est[0:N-1]) - (x_gt[1:N] - x_gt[0:N-1])) **2 + ((y_est[1:N] - y_est[0:N-1]) - (y_gt[1:N] - y_gt[0:N-1])) **2
            rpe_theta = (theta_est[1:N] - theta_est[0:N-1]) - (theta_gt[1:N] - theta_gt[0:N-1])
            rpe_ori[alg] += np.arctan2(np.sin(rpe_theta), np.cos(rpe_theta)) **2 
            
    data_num = robot_num * iter_num * N

    print('ALG:   APE_POS[m]    APE_ORI[rad]   RPE_POS[m]    RPE_ORI[rad]    NEES_POS     NEES_ORI ')

    for alg in algorithms:
      print('{}: {} {} {} {} {} {}'.format(alg, np.sqrt(np.sum(rmse_pos[alg]) / data_num), np.sqrt(np.sum(rmse_ori[alg]) / data_num), np.sqrt(np.sum(rpe_pos[alg]) / (data_num-1)), np.sqrt(np.sum(rpe_ori[alg]) / (data_num -1 )), np.sum(nees_pos[alg]) / data_num, np.sum(nees_ori[alg]) / data_num))

    N_step = int(N / 20)

    # save_path_fig = '../sim_results' + '/' + t + '/Figures' + '/'

    plt_rmse_pos = plt.figure(figsize=(8,4))
    ax1 = plt.subplot(211)
    # ax1 = plt.gca()
    # _set_axis(ax1)
    plt.xlabel('t (s)', fontdict={'family' : 'Times New Roman', 'size'   : 14})
    plt.ylabel(r'$\rm Pos. \ (m)$', fontdict={'family' : 'Times New Roman', 'size'   : 14})
    # plt.title('RMSE of position and orientation', fontsize=18)
   
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    # ax1.tick_params(axis='both', labelsize=14)
    plt.ylim((0, 8))

    # plt_rmse_psi = plt.figure(figsize=(6,2))
    ax2 = plt.subplot(212)
    # ax2 = plt.gca()
    # _set_axis(ax2)
    plt.xlabel('t (s)', fontdict={'family' : 'Times New Roman', 'size'   : 14})
    plt.ylabel(r'$\rm Ori. \ (rad)$', fontdict={'family' : 'Times New Roman', 'size'   : 14})

    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.ylim((0, 0.9))

    # ax2.tick_params(axis='both', labelsize=14)

    for alg in algorithms:
      # if alg != 'ekf':
        ax1.plot(time_arr[range(0, N, N_step)], np.sqrt(rmse_pos[alg] / robot_num / iter_num)[range(0, N, N_step)], color=color_tables[alg], label=label_tables[alg], linestyle=style_table[alg], linewidth=0.8, marker = marker_tables[alg], markerfacecolor='none', markersize=4)
        ax2.plot(time_arr[range(0, N, N_step)], np.sqrt(rmse_ori[alg] / robot_num / iter_num)[range(0, N, N_step)], color=color_tables[alg], label=label_tables[alg], linestyle=style_table[alg], linewidth=0.8, marker = marker_tables[alg], markerfacecolor='none', markersize=4)       
      #  color_tables[alg]

    ax1.legend(loc = 'upper left', frameon=False, ncol = 2, prop = {'family':'Times New Roman', 'size':14})
    ax2.legend(loc = 'upper left', frameon=False, ncol = 2, prop = {'family':'Times New Roman', 'size':14})

    current_path = os.getcwd()
    plt_rmse_pos.savefig(current_path + "/figures/pos_rmse" + '.png', dpi=600, bbox_inches='tight')
    # plt_rmse_psi.savefig(current_path + "/figures/psi_rmse" + '.png', dpi=600, bbox_inches='tight')
  
    plt_nees_pos = plt.figure(figsize=(8,4))
    # ax3 = plt.gca()
    ax3 = plt.subplot(211)
    plt.xlabel('t (s)', fontdict={'family' : 'Times New Roman', 'size'   : 14})
    # plt.ylabel(r'$\rm p \ \ log(NEES)$', fontsize=12)
    plt.ylabel(r'$\rm Pos.$', fontdict={'family' : 'Times New Roman', 'size'   : 14})
    # ax3.tick_params(axis='both', labelsize=14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)

    # plt_nees_psi = plt.figure(figsize=(6,2))
    ax4 = plt.subplot(212)
    # _set_axis(ax4)
    plt.xlabel('t (s)', fontdict={'family' : 'Times New Roman', 'size'   : 14})
    # plt.ylabel(r'$\rm \psi \ \ log(NEES)$', fontsize=12)
    plt.ylabel(r'$\rm Ori.$', fontdict={'family' : 'Times New Roman', 'size'   : 14})
    # ax4.tick_params(axis='both', labelsize=14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)

    for alg in algorithms:
      # if alg != 'ekf':
      # ax3.plot(time_arr[range(1, N, N_step)], np.log(nees_pos[alg] / robot_num / iter_num + 1)[range(1, N, N_step)], color=color_tables[alg], label=label_tables[alg], linestyle=style_table[alg], linewidth=1.8, marker = marker_tables[alg], markerfacecolor=color_tables[alg], markersize=4)         
      # ax4.plot(time_arr[range(1, N, N_step)], np.log(nees_ori[alg] / robot_num / iter_num + 1)[range(1, N, N_step)], color=color_tables[alg], label=label_tables[alg], linestyle=style_table[alg], linewidth=1.8, marker = marker_tables[alg], markerfacecolor=color_tables[alg], markersize=4)         
      ax3.plot(time_arr[range(1, N, N_step)], (nees_pos[alg] / robot_num / iter_num)[range(1, N, N_step)], color=color_tables[alg], label=label_tables[alg], linestyle=style_table[alg], linewidth=0.8, marker = marker_tables[alg], markerfacecolor='none', markersize=4)         
      ax4.plot(time_arr[range(1, N, N_step)], (nees_ori[alg] / robot_num / iter_num)[range(1, N, N_step)], color=color_tables[alg], label=label_tables[alg], linestyle=style_table[alg], linewidth=0.8, marker = marker_tables[alg], markerfacecolor='none', markersize=4)         
  
    ax3.legend(loc = 'upper left', frameon=False, ncol = 2, prop = {'family':'Times New Roman', 'size':14})
    ax4.legend(loc = 'upper left', frameon=False, ncol = 2, prop = {'family':'Times New Roman', 'size':14})

    # ax3.axhline(2, linestyle='--', linewidth=0.5)
    # ax4.axhline(1, linestyle='--', linewidth=0.5)

    current_path = os.getcwd()
    plt_nees_pos.savefig(current_path + "/figures/pos_nees" + '.png', dpi=600, bbox_inches='tight')
    # plt_nees_psi.savefig(current_path + "/figures/psi_nees" + '.png', dpi=600, bbox_inches='tight')
  
    # box-plot (position and orientation rmse)
    plt_rmse3 = plt.figure(figsize=(6, 4))
    # plt_rmse_ax3 = plt.gca()
    plt_rmse_ax3 = plt.subplot(211)
    plt.ylabel('Pos. (m)', fontdict={'family' : 'Times New Roman', 'size'   : 14})
    # plt_rmse_ax3.tick_params(axis='both', labelsize=14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)

    # plt_rmse4 = plt.figure(figsize=(6, 4))
    # plt_rmse_ax4 = plt.gca()
    plt_rmse_ax4 = plt.subplot(212)
    plt.ylabel(r'$\rm Ori. \ (rad)$', fontdict={'family' : 'Times New Roman', 'size'   : 14})
    # plt_rmse_ax4.tick_params(axis='both', labelsize=14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)

    data_pos = []
    data_ori = []
    labels = []
    colors = []

    for alg in algorithms:
        
        # for r in range(NUM_ROBOTS):
            # if r == 0:
            #     pos_rmse = (x_est_[alg][str(r+1)] - x_gt_[alg][str(r+1)])**2 + (y_est_[alg][str(r+1)] - y_gt_[alg][str(r+1)])**2
            #     ori_rmse = (np.arctan2(np.sin(theta_est_[alg][str(r+1)] - theta_gt_[alg][str(r+1)]), np.cos(theta_est_[alg][str(r+1)] - theta_gt_[alg][str(r+1)]))) ** 2
            # else:
            #     pos_rmse += (x_est_[alg][str(r+1)] - x_gt_[alg][str(r+1)])**2 + (y_est_[alg][str(r+1)] - y_gt_[alg][str(r+1)])**2
            #     ori_rmse += (np.arctan2(np.sin(theta_est_[alg][str(r+1)] - theta_gt_[alg][str(r+1)]), np.cos(theta_est_[alg][str(r+1)] - theta_gt_[alg][str(r+1)]))) ** 2

        pos_rmse = np.sqrt(rmse_pos[alg] / robot_num / iter_num)
        # rmse_pos[alg]
        ori_rmse = np.sqrt(rmse_ori[alg] / robot_num / iter_num)
        # rmse_ori[alg]

        # print(alg,  utias_ori_rmse[alg])

        # print(pos_rmse.shape[0])
        print('{}: {:.4f}/{:.4f}'.format(alg, np.sum(pos_rmse) / pos_rmse.shape[0], np.sum(ori_rmse) / ori_rmse.shape[0]))

        data_pos.append(pos_rmse)
        data_ori.append(ori_rmse)
        labels.append(label_tables[alg])
        colors.append(color_tables[alg])

    alg = 'ekf'

    # color_tables[alg]
    mean = {'linestyle':'-','color':color_tables[alg]}

    median = {'linestyle':'--','color':'purple'}

    showfilter = False
    shownortch = True

    # bplot_pos = plt_rmse_ax3.boxplot(data_pos, labels=labels, meanline=True, showmeans=True, showfliers= showfilter, notch = shownortch, vert=True, patch_artist=False, boxprops=box, meanprops=mean, medianprops = median, capprops = cap, whiskerprops = whisker, flierprops = flier)
    # bplot_ori = plt_rmse_ax4.boxplot(data_ori, labels=labels, meanline=True, showmeans=True, showfliers= showfilter, notch = shownortch, vert=True, patch_artist=False, boxprops=box, meanprops=mean, medianprops = median, capprops = cap, whiskerprops = whisker, flierprops = flier)
    bplot_pos = plt_rmse_ax3.boxplot(data_pos, notch=True, widths = 0.2, vert=True, showfliers=False, showmeans=True, labels=labels)
    # bplot_pos = plt_rmse_ax3.boxplot(data_pos, notch=True, vert=True, labels=None, showfliers= showfilter, flierprops = flier, boxprops=box,  capprops = cap, showmeans=True, meanline=True, meanprops=mean, whiskerprops = whisker)
    bplot_ori = plt_rmse_ax4.boxplot(data_ori, notch=True, widths = 0.2, vert=True, showfliers=False, showmeans=True, labels=labels)
    # bplot_ori = plt_rmse_ax4.boxplot(data_ori, notch=True, vert=True, labels=None, showfliers= showfilter, flierprops = flier, boxprops=box,  capprops = cap, showmeans=True, meanline=True, meanprops=mean, whiskerprops = whisker)

    for alg in algorithms:
      # bplot_pos['boxes'][algorithms.index(alg)].set_facecolor(color_tables[alg])
      bplot_pos['boxes'][algorithms.index(alg)].set_color(color_tables[alg])
      bplot_ori['boxes'][algorithms.index(alg)].set_color(color_tables[alg])
      
      for i in range(2):
        bplot_pos['caps'][2*algorithms.index(alg)].set_color(color_tables[alg])
        bplot_pos['caps'][2*algorithms.index(alg)+1].set_color(color_tables[alg])
        bplot_ori['caps'][2*algorithms.index(alg)].set_color(color_tables[alg])
        bplot_ori['caps'][2*algorithms.index(alg)+1].set_color(color_tables[alg])
      
        bplot_pos['whiskers'][2*algorithms.index(alg)].set_color(color_tables[alg])
        bplot_pos['whiskers'][2*algorithms.index(alg)+1].set_color(color_tables[alg])
        bplot_ori['whiskers'][2*algorithms.index(alg)].set_color(color_tables[alg])
        bplot_ori['whiskers'][2*algorithms.index(alg)+1].set_color(color_tables[alg])

    bplot_ori['means'][algorithms.index(alg)].set_color(color_tables[alg])

    # print(bplot_pos)
    
    # print(len(bplot_pos['caps']))
    # print(algorithms.index('kdp'))

    # plt_rmse_ax3.legend(bplot_pos['boxes'], labels, frameon=True, ncol = 4, loc='upper right')

    # plt_rmse_ax3.axhline(np.mean(data_pos[algorithms.index('kdp')]), linestyle='--', linewidth=0.5)
    # plt_rmse_ax4.axhline(np.mean(data_ori[algorithms.index('kdp')]), linestyle='--', linewidth=0.5)

    # plt_rmse_ax3.axhline(np.median(data_pos[algorithms.index('kdl')]), linestyle='--', linewidth=0.5)
    # plt_rmse_ax4.axhline(np.median(data_ori[algorithms.index('kdl')]), linestyle='--', linewidth=0.5)

    # plt_rmse_ax3.legend()

    # colors = ['pink', 'lightblue', 'lightgreen']  ##定义柱子颜色、和柱子数目一致

    # for patch, color in zip(bplot_pos['boxes'], colors): ##zip快速取出两个长度相同的数组对应的索引值
    #     patch.set_color(color) 

    # for patch, color in zip(bplot_ori['boxes'], colors): ##zip快速取出两个长度相同的数组对应的索引值
    #     patch.set_color(color) 


    current_path = os.getcwd()
    plt_rmse3.savefig(current_path + "/figures/psi_pos_rmse_b" + '.png', dpi=600, bbox_inches='tight')
  
    plt_traj = plt.figure(figsize=(6, 4))
    plt_traj_ax = plt.gca()
    plt.xlabel('x (m)', fontdict={'family' : 'Times New Roman', 'size'   : 14})
    plt.ylabel('y (m)', fontdict={'family' : 'Times New Roman', 'size'   : 14})
    # plt_traj_ax.tick_params(axis='both', labelsize=14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)

    save_path = '../sim_results' + '/' + t + '/MRCLAM' + str(1) + '/'

    style_list = ['-', '--', '-.', ':']
    color_list = ['orange', 'blue', 'red', 'lime']

    for r in range(NUM_ROBOTS):
      file_name = 'ekf' + "_robot" + str(r+1) + '.npz'
      data = np.load(save_path + file_name)

      x_gt = data['x_gt']
      y_gt = data['y_gt']

      plt_traj_ax.plot(x_gt, y_gt, color=color_list[r], label='robot'+str(r+1), linewidth=1)

    plt_traj_ax.legend(loc = 'upper left', frameon=True, ncol = 2, prop = {'size':10})
    
    current_path = os.getcwd()
    plt_traj.savefig(current_path + "/figures/traj" + '.png', dpi=600, bbox_inches='tight')
  

    plt_rpe_pos = plt.figure(figsize=(8,4))
    ax1_rpe = plt.subplot(211)
    # ax1_rpe = plt.gca()
    # _set_axis(ax1)
    plt.xlabel('t (s)', fontdict={'family' : 'Times New Roman', 'size'   : 14})
    plt.ylabel(r'$\rm Pos. \ (m)$', fontdict={'family' : 'Times New Roman', 'size'   : 14})
    # plt.title('RMSE of position and orientation', fontsize=18)
    # ax1_rpe.tick_params(axis='both', labelsize=14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)

    # plt_rpe_psi = plt.figure(figsize=(6,2))
    ax2_rpe = plt.subplot(212)
    # ax2_rpe = plt.gca()
    # _set_axis(ax2)
    plt.xlabel('t (s)', fontdict={'family' : 'Times New Roman', 'size'   : 14})
    plt.ylabel(r'$\rm Ori. \ (rad)$', fontdict={'family' : 'Times New Roman', 'size'   : 14})
    # ax2_rpe.tick_params(axis='both', labelsize=14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)
    plt.ylim((0, 0.4))

    for alg in algorithms:
      # if alg != 'ekf':
        ax1_rpe.plot(time_arr[range(0, N-1, N_step)], np.sqrt(rpe_pos[alg] / robot_num / iter_num)[range(0, N-1, N_step)], color=color_tables[alg], label=label_tables[alg], linestyle=style_table[alg], linewidth=0.8, marker = marker_tables[alg], markerfacecolor='none', markersize=4)
        ax2_rpe.plot(time_arr[range(0, N-1, N_step)], np.sqrt(rpe_ori[alg] / robot_num / iter_num)[range(0, N-1, N_step)], color=color_tables[alg], label=label_tables[alg], linestyle=style_table[alg], linewidth=0.8, marker = marker_tables[alg], markerfacecolor='none', markersize=4)       
 
    ax1_rpe.legend(loc = 'upper left', frameon=False, ncol = 2, prop = {'family':'Times New Roman', 'size':14})
    ax2_rpe.legend(loc = 'upper left', frameon=False, ncol = 2, prop = {'family':'Times New Roman', 'size':14})

    current_path = os.getcwd()
    plt_rpe_pos.savefig(current_path + "/figures/pos_rpe" + '.png', dpi=600, bbox_inches='tight')
    # plt_rpe_psi.savefig(current_path + "/figures/psi_rpe" + '.png', dpi=600, bbox_inches='tight')


    plt_rpe3 = plt.figure(figsize=(6, 4))
    # plt_rmse_ax3 = plt.gca()
    plt_rpe_ax3 = plt.subplot(211)
    plt.ylabel('Pos. (m)', fontdict={'family' : 'Times New Roman', 'size'   : 14})
    # plt_rpe_ax3.tick_params(axis='both', labelsize=14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)

    # plt_rmse4 = plt.figure(figsize=(6, 4))
    # plt_rmse_ax4 = plt.gca()
    plt_rpe_ax4 = plt.subplot(212)
    plt.ylabel(r'$\rm Ori. \ (rad)$', fontdict={'family' : 'Times New Roman', 'size'   : 14})
    # plt_rpe_ax4.tick_params(axis='both', labelsize=14)
    plt.yticks(fontproperties = 'Times New Roman', size = 14)
    plt.xticks(fontproperties = 'Times New Roman', size = 14)

    data_pos = []
    data_ori = []
    labels = []
    colors = []

    for alg in algorithms:
        
        pos_rpe = np.sqrt(rpe_pos[alg] / robot_num / iter_num)
        # rmse_pos[alg]
        ori_rpe = np.sqrt(rpe_ori[alg] / robot_num / iter_num)
        # rmse_ori[alg]

        # print(alg,  utias_ori_rmse[alg])

        # print(pos_rmse.shape[0])
        print('{}: {:.4f}/{:.4f}'.format(alg, np.sum(pos_rpe) / pos_rpe.shape[0], np.sum(ori_rpe) / ori_rpe.shape[0]))

        data_pos.append(pos_rpe)
        data_ori.append(ori_rpe)
        labels.append(label_tables[alg])
        colors.append(color_tables[alg])

    alg = 'ekf'

    # color_tables[alg]
    mean = {'linestyle':'-','color':color_tables[alg]}

    median = {'linestyle':'--','color':'purple'}

    showfilter = False
    shownortch = True

    # bplot_pos = plt_rmse_ax3.boxplot(data_pos, labels=labels, meanline=True, showmeans=True, showfliers= showfilter, notch = shownortch, vert=True, patch_artist=False, boxprops=box, meanprops=mean, medianprops = median, capprops = cap, whiskerprops = whisker, flierprops = flier)
    # bplot_ori = plt_rmse_ax4.boxplot(data_ori, labels=labels, meanline=True, showmeans=True, showfliers= showfilter, notch = shownortch, vert=True, patch_artist=False, boxprops=box, meanprops=mean, medianprops = median, capprops = cap, whiskerprops = whisker, flierprops = flier)
    bplot_pos = plt_rpe_ax3.boxplot(data_pos, notch=True, widths = 0.2, vert=True, showfliers=False, showmeans=True, labels=labels)
    # bplot_pos = plt_rmse_ax3.boxplot(data_pos, notch=True, vert=True, labels=None, showfliers= showfilter, flierprops = flier, boxprops=box,  capprops = cap, showmeans=True, meanline=True, meanprops=mean, whiskerprops = whisker)
    bplot_ori = plt_rpe_ax4.boxplot(data_ori, notch=True, widths = 0.2, vert=True, showfliers=False, showmeans=True, labels=labels)
    # bplot_ori = plt_rmse_ax4.boxplot(data_ori, notch=True, vert=True, labels=None, showfliers= showfilter, flierprops = flier, boxprops=box,  capprops = cap, showmeans=True, meanline=True, meanprops=mean, whiskerprops = whisker)

    for alg in algorithms:
      # bplot_pos['boxes'][algorithms.index(alg)].set_facecolor(color_tables[alg])
      bplot_pos['boxes'][algorithms.index(alg)].set_color(color_tables[alg])
      bplot_ori['boxes'][algorithms.index(alg)].set_color(color_tables[alg])
      
      for i in range(2):
        bplot_pos['caps'][2*algorithms.index(alg)].set_color(color_tables[alg])
        bplot_pos['caps'][2*algorithms.index(alg)+1].set_color(color_tables[alg])
        bplot_ori['caps'][2*algorithms.index(alg)].set_color(color_tables[alg])
        bplot_ori['caps'][2*algorithms.index(alg)+1].set_color(color_tables[alg])
      
        bplot_pos['whiskers'][2*algorithms.index(alg)].set_color(color_tables[alg])
        bplot_pos['whiskers'][2*algorithms.index(alg)+1].set_color(color_tables[alg])
        bplot_ori['whiskers'][2*algorithms.index(alg)].set_color(color_tables[alg])
        bplot_ori['whiskers'][2*algorithms.index(alg)+1].set_color(color_tables[alg])

    bplot_ori['means'][algorithms.index(alg)].set_color(color_tables[alg])

    # plt_rpe_ax3.axhline(np.median(data_pos[algorithms.index('kdl')]), linestyle='--', linewidth=0.5)
    # plt_rpe_ax4.axhline(np.median(data_ori[algorithms.index('kdl')]), linestyle='--', linewidth=0.5)

    current_path = os.getcwd()
    plt_rpe3.savefig(current_path + "/figures/psi_pos_rpe_b" + '.png', dpi=600, bbox_inches='tight')

    plt.show()
