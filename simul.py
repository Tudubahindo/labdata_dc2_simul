#!/usr/bin/env python

import matplotlib as mpl
#mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform

def hopkins_statistic(X):
    
    X=X.values  #convert dataframe to a numpy array
    sample_size = int(X.shape[0]*0.05) #0.05 (5%) based on paper by Lawson and Jures
    
    
    #a uniform random sample in the original data space
    X_uniform_random_sample = uniform(X.min(axis=0), X.max(axis=0) ,(sample_size , X.shape[1]))
    
    
    
    #a random sample of size sample_size from the original data X
    random_indices=sample(range(0, X.shape[0], 1), sample_size)
    X_sample = X[random_indices]
   
    
    #initialise unsupervised learner for implementing neighbor searches
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs=neigh.fit(X)
    
    #u_distances = nearest neighbour distances from uniform random sample
    u_distances , u_indices = nbrs.kneighbors(X_uniform_random_sample , n_neighbors=2)
    u_distances = u_distances[: , 0] #distance to the first (nearest) neighbour
    
    #w_distances = nearest neighbour distances from a sample of points from original data X
    w_distances , w_indices = nbrs.kneighbors(X_sample , n_neighbors=2)
    #distance to the second nearest neighbour (as the first neighbour will be the point itself, with distance = 0)
    w_distances = w_distances[: , 1]
    
 
    
    u_sum = np.sum(u_distances)
    w_sum = np.sum(w_distances)
    
    #compute and return hopkins' statistic
    H = u_sum/ (u_sum + w_sum)
    return H

#CALIBRAZIONE HOPKINS
#sigma positivo: cluster; negativo: repulsione; nullo: PPP
#usate sigma interi. Più i sigma sono grandi, in valore assoluto, più vi allontanate dal processo stocastico

def hopkins_calibration(N=1000, rho=0):

    N = int(N)      #Prelz dice sempre: venite incontro all'utente scemo. cast double in int
    v = np.zeros(N)

    #processo stocastico: PPP
    if rho == 0:
        v = np.random.uniform(0,1,N)
    #Gaussiana centrata in 0.5, larghezza inversamente proporzionale a sigma
    if rho > 0:
        stdev = 1/rho
        v = np.random.normal(0.5,stdev,N)
        for i in range(N):
            while v[i]<0 or v[i]>1:
                v[i] = np.random.normal(0.5,stdev,1)
    #Dati equispaziati, con rumore gaussiano. Maggiore sigma (in modulo), più piccolo il rumore
    if rho < 0:
        delta = 1/N
        stdev = delta/np.abs(rho)
        dh = delta/2
        for i in range(N):
            v[i] = np.random.normal(dh + i*delta,stdev,1) 

    return v

#BARCODE PLOTS

def dataset_visualization():
    
    n = 100
    v0 = hopkins_calibration(n,0)
    v1 = hopkins_calibration(n,10)
    v2 = hopkins_calibration(n,-10)
    #print(v0)
    
    l = 400
    code0 = np.zeros(l)
    for i in range(l):
        for j in range(n):
            if v0[j] > i/l and v0[j] < (i+1)/l :
                code0[i]=1
    code1 = np.zeros(l)
    for i in range(l):
        for j in range(n):
            if v1[j] > i/l and v1[j] < (i+1)/l :
                code1[i]=1
    code2 = np.zeros(l)
    for i in range(l):
        for j in range(n):
            if v2[j] > i/l and v2[j] < (i+1)/l :
                code2[i]=1
    
    pixel_per_bar = 4
    dpi = 100
    
    
    
    fig0 = plt.figure(figsize=(len(code0) * pixel_per_bar / dpi, 2), dpi=dpi)
    ax0 = fig0.add_axes([0, 0, 1, 1])  # span the whole figure
    ax0.set_axis_off()
    ax0.imshow(code0.reshape(1, -1), cmap='binary', aspect='auto',
              interpolation='nearest')

    fig1 = plt.figure(figsize=(len(code1) * pixel_per_bar / dpi, 2), dpi=dpi)
    ax1 = fig1.add_axes([0, 0, 1, 1])  # span the whole figure
    ax1.set_axis_off()
    ax1.imshow(code1.reshape(1, -1), cmap='binary', aspect='auto',
              interpolation='nearest')

    fig2 = plt.figure(figsize=(len(code2) * pixel_per_bar / dpi, 2), dpi=dpi)
    ax2 = fig2.add_axes([0, 0, 1, 1])  # span the whole figure
    ax2.set_axis_off()
    ax2.imshow(code2.reshape(1, -1), cmap='binary', aspect='auto',
              interpolation='nearest')

def sogliole(num = 1000, simulnum = 10000, rho = 0, printing = False):
    
    #CALIBRAZIONE SOGLIOLE
    
    H = np.zeros(simulnum)
    stdevrel = np.zeros(simulnum)
    for i in range(simulnum):
        v = hopkins_calibration(num,rho)
        v = np.sort(v)
        H[i]=hopkins_statistic(pd.DataFrame(v))
        length = len(v) - 1
        inter = np.zeros(length)
        for j in range(length):
            inter[j] = v[j+1] - v[j]
        stdevrel[i] = np.std(inter)/np.mean(inter)
    
    H = np.sort(H)
    H_splice = H[int(simulnum*0.025 + 1):int(simulnum*0.975)]
    
    stdevrel = np.sort(stdevrel)
    stdevrel_splice = stdevrel[int(simulnum*0.025 + 1):int(simulnum*0.975)]
    
    if printing == True:
        print("H min: ", np.min(H), "\t H mean: ", np.mean(H), "\t H max: ", np.max(H))
        print("H 2.5pct: ", np.min(H_splice), "\t H 97.5pct: ", np.max(H_splice))
        print("S min: ", np.min(stdevrel), "\t S mean: ", np.mean(stdevrel), "\t S max: ", np.max(stdevrel))
        print("S 2.5pct: ", np.min(stdevrel_splice), "\t S 97.5pct: ", np.max(stdevrel_splice))
    
    return np.mean(H)

#MAIN

def main():
    dataset_visualization()
    sogliole(rho=0, printing=True)
    
    index = np.zeros(21)
    hoprho = np.zeros(21)
    for i in range(21):
        index[i]=i-10
        hoprho[i]=sogliole(rho=i-10)
        
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    ax1.scatter(index[0:10],hoprho[0:10],color='red',s=480, label='repulsione')
    ax1.scatter(index[10],hoprho[10],color='green',marker='o', s=480, label='non interazione')
    ax1.scatter(index[11:21],hoprho[11:21],color='blue',marker='o', s=480, label='attrazione')
    ax1.set_xlabel('ρ', fontsize=30)
    ax1.set_ylabel('Statistica di Hopkins', fontsize=30)
    ax1.tick_params(axis = 'both', labelsize = 25)
    ax1.set_axisbelow(True)
    ax1.grid(axis='both', linewidth=2)
    ax1.legend(fontsize=30)
     
    df = pd.read_csv('dati_puliti.csv',delimiter=',')

    (dfTT, dfG, dfM, dfN, dfA, dfT) = [pd.DataFrame(df.iloc[:,i]).dropna() for i in range(6)]
    
    dfD = pd.read_csv('dataDante.dat')
    dfD = pd.DataFrame(dfD)
   
    dfPT = pd.read_csv('place_tux.dat')
    dfPT = pd.DataFrame(dfPT)
    
    dfPC = pd.read_csv('place_colosseo.dat')
    dfPC = pd.DataFrame(dfPC)

    H_T = 0
    H_G = 0
    H_M = 0
    H_N = 0
    H_A = 0
    H_D = 0
    H_PT = 0
    H_PC = 0

    for i in range(100):
        H_T += hopkins_statistic(dfT)
        H_G += hopkins_statistic(dfG)
        H_M += hopkins_statistic(dfM)
        H_N += hopkins_statistic(dfN)
        H_A += hopkins_statistic(dfA)
        H_D += hopkins_statistic(dfD)
        H_PT += hopkins_statistic(dfPT)
        H_PC += hopkins_statistic(dfPC)
        
    H_T /= 100
    H_G /= 100
    H_M /= 100
    H_N /= 100
    H_A /= 100
    H_D /= 100
    H_PT /= 100
    H_PC /= 100
    
    print('H terremoti: ', H_T, '\nH geni: ', H_G, '\nH mexicodoug: ', H_M, '\nH nixonrichard: ', H_N,'\nH aletoledo: ', H_A, '\nH Commedia: ', H_D, '\nH tux: ', H_PT, '\nH colosseo: ', H_PC)

    V_G = np.zeros(len(dfG))
    V_D = np.zeros(len(dfD))
    V_T = np.zeros(len(dfT))
    
    for i in range(len(dfG)):
        V_G[i] = dfG.iloc[i,0]
    for i in range(len(dfD)):
        V_D[i] = dfD.iloc[i,0]
    for i in range(len(dfT)):
        V_T[i] = dfT.iloc[i,0]

    V_G = np.sort(V_G)
    inter_G = np.zeros(len(V_G)-1)
    for j in range(len(V_G)-1):
        inter_G[j] = V_G[j+1] - V_G[j]
    V_D = np.sort(V_D)
    inter_D = np.zeros(len(V_D)-1)
    for j in range(len(V_D)-1):
        inter_D[j] = V_D[j+1] - V_D[j]
    V_T = np.sort(V_T)
    inter_T = np.zeros(len(V_T)-1)
    for j in range(len(V_T)-1):
        inter_T[j] = V_T[j+1] - V_T[j]

    d_G = sp.stats.anderson(inter_G, dist='expon')
    d_D = sp.stats.anderson(inter_D, dist='expon')
    d_T = sp.stats.anderson(inter_T, dist='expon')

    print('Geni: ',d_G,'\n Terremoti: ', d_T,'\n Commedia: ', d_D,'\n')

    stdev_G_rel = np.std(inter_G)/np.mean(inter_G)

    print('sigma geni: ', stdev_G_rel)
    
    lw=10

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
    ax2.axvline(0.4,color='grey',linestyle='-',linewidth=lw)
    ax2.axvline(0.6,color='grey',linestyle='-',linewidth=lw)
    ax2.tick_params(axis = 'x', labelsize = 25)
    ax2.tick_params(axis = 'y', labelsize = 0)
    ax2.axvline(H_G,color='green',linestyle='-',label='geni',linewidth=lw)
    ax2.axvline(H_T,color='orange',linestyle='-',label='terremoti',linewidth=lw)
    ax2.axvline(H_D,color='blue',linestyle='-',label='Commedia',linewidth=lw)
    ax2.axvline(H_N,color='red',linestyle='-',label='reddit',linewidth=lw)
    ax2.axvline(H_M,color='red',linestyle='-',linewidth=lw)
    ax2.axvline(H_A,color='red',linestyle='-',linewidth=lw)
    ax2.axvline(H_PC,color='violet',linestyle='-',label='r/place',linewidth=lw)
    ax2.axvline(H_PT,color='violet',linestyle='-',linewidth=lw)
    ax2.set_xlabel('Hopkins', fontsize=30)
    ax2.legend(fontsize=30)
    
    plt.show()
 
if __name__ == '__main__':
    main()
 
