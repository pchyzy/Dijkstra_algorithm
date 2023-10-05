#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.colors as colors
import pandas as pd
import re
import MDAnalysis as mda

def hex_to_rgb(value):
    value = value.strip("#")
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_dec(value):
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

def Dijkstra(input_name,max_PMF,number_of_bins,bins_X,bins_Y,cm_X,cm_Y,error,CV_input,PDB_input,DCD_input,group_atom,output_name):
    pmf = pd.read_csv(r""+str(input_name)+"",sep=' ',names=['First','Second','Value'])
    pmf.loc[pmf['Value'] >= max_PMF, 'Value'] = max_PMF
    max_val = number_of_bins
    new_pmf = np.zeros(shape=(max_val,max_val))
    for i in range(len(pmf)):
        row, column = (max_val-1)-int(i/max_val), (max_val-1)-int(i%max_val)
        new_pmf[row][column] += pmf['Value'][i]
        
    map = new_pmf
    distmap = np.ones((max_val,max_val),dtype=int)*np.Infinity
    originmap = np.ones((max_val,max_val),dtype=int)*np.nan
    visited = np.zeros((max_val,max_val),dtype=bool)
    distmap[0,0]=0
    finished = False
    x,y,count=np.int(0),np.int(0),0
    
    while not finished:
      if x < max_val-1:
        if distmap[x+1,y]>map[x+1,y]+distmap[x,y] and not visited[x+1,y]:
          distmap[x+1,y]=map[x+1,y]+distmap[x,y]
          originmap[x+1,y]=np.ravel_multi_index([x,y], (max_val,max_val))
      if x>0:
        if distmap[x-1,y]>map[x-1,y]+distmap[x,y] and not visited[x-1,y]:
          distmap[x-1,y]=map[x-1,y]+distmap[x,y]
          originmap[x-1,y]=np.ravel_multi_index([x,y], (max_val,max_val))
      if y < max_val-1:
        if distmap[x,y+1]>map[x,y+1]+distmap[x,y] and not visited[x,y+1]:
          distmap[x,y+1]=map[x,y+1]+distmap[x,y]
          originmap[x,y+1]=np.ravel_multi_index([x,y], (max_val,max_val))
      if y>0:
        if distmap[x,y-1]>map[x,y-1]+distmap[x,y] and not visited[x,y-1]:
          distmap[x,y-1]=map[x,y-1]+distmap[x,y]
          originmap[x,y-1]=np.ravel_multi_index([x,y], (max_val,max_val))
    
      visited[x,y]=True
      dismaptemp=distmap
      dismaptemp[np.where(visited)]=np.Infinity
      minpost=np.unravel_index(np.argmin(dismaptemp),np.shape(dismaptemp))
      x,y=minpost[0],minpost[1]
      if x==max_val-1 and y==max_val-1:
        finished=True
      count=count+1
     
    mattemp=map.astype(float)
    x,y,path=max_val-1,max_val-1,[]
    mattemp[np.int(x),np.int(y)]=np.nan
    
    while x>0.0 or y>0.0:
      path.append([np.int(x),np.int(y)])
      xxyy=np.unravel_index(np.int(originmap[np.int(x),np.int(y)]), (max_val,max_val))
      x,y=xxyy[0],xxyy[1]
      mattemp[np.int(x),np.int(y)]=np.nan
    path.append([np.int(x),np.int(y)])
    
    path_x, path_y = [], []
    for i in path:
        path_x.append(int(i[0]))
        path_y.append(int(i[1]))
        
    x_transformed, y_transformed = [], []
    for i in range(len(path_x)):
        valx = bins_X[(max_val-1)-path_x[i]]
        x_transformed.append(valx)
        valy = bins_Y[(max_val-1)-path_y[i]]
        y_transformed.append(valy)
        
    x_value_raw, y_value_raw = pmf['First'].tolist(), pmf['Second'].tolist()
    x_value, y_value = [], []
    for i in x_value_raw:
        if i not in x_value:
            if i <= 100:
                x_value.append(i)
                
    for i in y_value_raw:
        if i not in y_value:
            if i <= 100:
                y_value.append(i)
        
    z_value = (np.array(pmf['Value'].tolist())).reshape(len(x_value),len(y_value))
    hex_list = ['#ffffff','#000000','#000000','#ff0000','#ff8800','#ffea00','#48ff00','#002aff','#002aff','#00fff7','#cffffc']
    Y, X = np.meshgrid(y_value, x_value)

    plt.figure(figsize=((cm_X/2.54),(cm_Y/2.54)))
    interval = np.arange(0, max_PMF+1, 1)
    c1 = plt.contourf(X, Y, z_value, interval, cmap=get_continuous_cmap(hex_list))
    c2 = plt.contour(c1,interval,colors='k',linewidths=0.1)
    c = plt.colorbar(c1,pad=0.01)
    plt.plot(x_transformed,y_transformed,color='#f104b1',marker=',',markersize=0.5)    
    plt.xlabel('Distance [Å]')
    plt.ylabel('RMSD [Å]')
    plt.clim(0,max_PMF)
    plt.savefig(str(output_name)+'.png',dpi=300, bbox_inches='tight')

    frames_to_add = []
    for coord in range(len(x_transformed)):
        x = x_transformed[coord]
        y = y_transformed[coord]
        data = pd.read_csv(str(CV_input),sep='\t',names=['Frame','Distance','RMSD'])
        sigma_v = error
        sigma_d = error
        selected = data.loc[(data['Distance']<=(x+sigma_d))&
                            (data['Distance']>=(x-sigma_d))&
                            (data['RMSD']<=(y+sigma_v))&
                            (data['RMSD']>=(y-sigma_v))]
        frames = selected['Frame'].tolist()
        if len(frames)!=0:
            for frame in frames:
                frames_to_add.append(frame)

    frames_to_add.reverse()
    u = mda.Universe(str(PDB_input),str(DCD_input))
    ag = u.select_atoms(str(group_atom))
    ag.write(str(output_name)+'.pdb')
    ag.write(str(output_name)+'.dcd', frames=u.trajectory[frames_to_add])

if __name__ == "__main__":
    input_name = 'pmf.dat'
    max_PMF = 40
    number_of_bins = 300
    bins_X = [8.04, 8.12, 8.2, 8.28, 8.36, 8.44, 8.52, 8.6, 8.68, 8.76, 8.84, 8.92, 9.0, 9.08, 9.16, 9.24, 9.32, 9.4, 9.48, 9.56, 9.64, 9.72, 9.8, 9.88, 9.96, 10.04, 10.12, 10.2, 10.28, 10.36, 10.44, 10.52, 10.6, 10.68, 10.76, 10.84, 10.92, 11.0, 11.08, 11.16, 11.24, 11.32, 11.4, 11.48, 11.56, 11.64, 11.72, 11.8, 11.88, 11.96, 12.04, 12.12, 12.2, 12.28, 12.36, 12.44, 12.52, 12.6, 12.68, 12.76, 12.84, 12.92, 13.0, 13.08, 13.16, 13.24, 13.32, 13.4, 13.48, 13.56, 13.64, 13.72, 13.8, 13.88, 13.96, 14.04, 14.12, 14.2, 14.28, 14.36, 14.44, 14.52, 14.6, 14.68, 14.76, 14.84, 14.92, 15.0, 15.08, 15.16, 15.24, 15.32, 15.4, 15.48, 15.56, 15.64, 15.72, 15.8, 15.88, 15.96, 16.04, 16.12, 16.2, 16.28, 16.36, 16.44, 16.52, 16.6, 16.68, 16.76, 16.84, 16.92, 17.0, 17.08, 17.16, 17.24, 17.32, 17.4, 17.48, 17.56, 17.64, 17.72, 17.8, 17.88, 17.96, 18.04, 18.12, 18.2, 18.28, 18.36, 18.44, 18.52, 18.6, 18.68, 18.76, 18.84, 18.92, 19.0, 19.08, 19.16, 19.24, 19.32, 19.4, 19.48, 19.56, 19.64, 19.72, 19.8, 19.88, 19.96, 20.04, 20.12, 20.2, 20.28, 20.36, 20.44, 20.52, 20.6, 20.68, 20.76, 20.84, 20.92, 21.0, 21.08, 21.16, 21.24, 21.32, 21.4, 21.48, 21.56, 21.64, 21.72, 21.8, 21.88, 21.96, 22.04, 22.12, 22.2, 22.28, 22.36, 22.44, 22.52, 22.6, 22.68, 22.76, 22.84, 22.92, 23.0, 23.08, 23.16, 23.24, 23.32, 23.4, 23.48, 23.56, 23.64, 23.72, 23.8, 23.88, 23.96, 24.04, 24.12, 24.2, 24.28, 24.36, 24.44, 24.52, 24.6, 24.68, 24.76, 24.84, 24.92, 25.0, 25.08, 25.16, 25.24, 25.32, 25.4, 25.48, 25.56, 25.64, 25.72, 25.8, 25.88, 25.96, 26.04, 26.12, 26.2, 26.28, 26.36, 26.44, 26.52, 26.6, 26.68, 26.76, 26.84, 26.92, 27.0, 27.08, 27.16, 27.24, 27.32, 27.4, 27.48, 27.56, 27.64, 27.72, 27.8, 27.88, 27.96, 28.04, 28.12, 28.2, 28.28, 28.36, 28.44, 28.52, 28.6, 28.68, 28.76, 28.84, 28.92, 29.0, 29.08, 29.16, 29.24, 29.32, 29.4, 29.48, 29.56, 29.64, 29.72, 29.8, 29.88, 29.96, 30.04, 30.12, 30.2, 30.28, 30.36, 30.44, 30.52, 30.6, 30.68, 30.76, 30.84, 30.92, 31.0, 31.08, 31.16, 31.24, 31.32, 31.4, 31.48, 31.56, 31.64, 31.72, 31.8, 31.88, 31.96]
    bins_Y = [1.0208, 1.0625, 1.1042, 1.1458, 1.1875, 1.2292, 1.2708, 1.3125, 1.3542, 1.3958, 1.4375, 1.4792, 1.5208, 1.5625, 1.6042, 1.6458, 1.6875, 1.7292, 1.7708, 1.8125, 1.8542, 1.8958, 1.9375, 1.9792, 2.0208, 2.0625, 2.1042, 2.1458, 2.1875, 2.2292, 2.2708, 2.3125, 2.3542, 2.3958, 2.4375, 2.4792, 2.5208, 2.5625, 2.6042, 2.6458, 2.6875, 2.7292, 2.7708, 2.8125, 2.8542, 2.8958, 2.9375, 2.9792, 3.0208, 3.0625, 3.1042, 3.1458, 3.1875, 3.2292, 3.2708, 3.3125, 3.3542, 3.3958, 3.4375, 3.4792, 3.5208, 3.5625, 3.6042, 3.6458, 3.6875, 3.7292, 3.7708, 3.8125, 3.8542, 3.8958, 3.9375, 3.9792, 4.0208, 4.0625, 4.1042, 4.1458, 4.1875, 4.2292, 4.2708, 4.3125, 4.3542, 4.3958, 4.4375, 4.4792, 4.5208, 4.5625, 4.6042, 4.6458, 4.6875, 4.7292, 4.7708, 4.8125, 4.8542, 4.8958, 4.9375, 4.9792, 5.0208, 5.0625, 5.1042, 5.1458, 5.1875, 5.2292, 5.2708, 5.3125, 5.3542, 5.3958, 5.4375, 5.4792, 5.5208, 5.5625, 5.6042, 5.6458, 5.6875, 5.7292, 5.7708, 5.8125, 5.8542, 5.8958, 5.9375, 5.9792, 6.0208, 6.0625, 6.1042, 6.1458, 6.1875, 6.2292, 6.2708, 6.3125, 6.3542, 6.3958, 6.4375, 6.4792, 6.5208, 6.5625, 6.6042, 6.6458, 6.6875, 6.7292, 6.7708, 6.8125, 6.8542, 6.8958, 6.9375, 6.9792, 7.0208, 7.0625, 7.1042, 7.1458, 7.1875, 7.2292, 7.2708, 7.3125, 7.3542, 7.3958, 7.4375, 7.4792, 7.5208, 7.5625, 7.6042, 7.6458, 7.6875, 7.7292, 7.7708, 7.8125, 7.8542, 7.8958, 7.9375, 7.9792, 8.0208, 8.0625, 8.1042, 8.1458, 8.1875, 8.2292, 8.2708, 8.3125, 8.3542, 8.3958, 8.4375, 8.4792, 8.5208, 8.5625, 8.6042, 8.6458, 8.6875, 8.7292, 8.7708, 8.8125, 8.8542, 8.8958, 8.9375, 8.9792, 9.0208, 9.0625, 9.1042, 9.1458, 9.1875, 9.2292, 9.2708, 9.3125, 9.3542, 9.3958, 9.4375, 9.4792, 9.5208, 9.5625, 9.6042, 9.6458, 9.6875, 9.7292, 9.7708, 9.8125, 9.8542, 9.8958, 9.9375, 9.9792, 10.0208, 10.0625, 10.1042, 10.1458, 10.1875, 10.2292, 10.2708, 10.3125, 10.3542, 10.3958, 10.4375, 10.4792, 10.5208, 10.5625, 10.6042, 10.6458, 10.6875, 10.7292, 10.7708, 10.8125, 10.8542, 10.8958, 10.9375, 10.9792, 11.0208, 11.0625, 11.1042, 11.1458, 11.1875, 11.2292, 11.2708, 11.3125, 11.3542, 11.3958, 11.4375, 11.4792, 11.5208, 11.5625, 11.6042, 11.6458, 11.6875, 11.7292, 11.7708, 11.8125, 11.8542, 11.8958, 11.9375, 11.9792, 12.0208, 12.0625, 12.1042, 12.1458, 12.1875, 12.2292, 12.2708, 12.3125, 12.3542, 12.3958, 12.4375, 12.4792, 12.5208, 12.5625, 12.6042, 12.6458, 12.6875, 12.7292, 12.7708, 12.8125, 12.8542, 12.8958, 12.9375, 12.9792, 13.0208, 13.0625, 13.1042, 13.1458, 13.1875, 13.2292, 13.2708, 13.3125, 13.3542, 13.3958, 13.4375, 13.4792]
    cm_X = 14.368
    cm_Y = 11.591
    error = 0.005
    
    CV_input = '../../replica_all.cv'
    PDB_input = '../../../../md_nowat.pdb'
    DCD_input = '../../../../replica_all.dcd'
    group_atom = 'resid 1 to 28'
    output_name = 'PNAS'
    Dijkstra(input_name,max_PMF,number_of_bins,bins_X,bins_Y,cm_X,cm_Y,error,CV_input,PDB_input,DCD_input,group_atom,output_name)

