# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 21:46:05 2017

@author: wuyuhang
"""
import random

'''
Calculate the probability of continuous lose of a team in a regular season. The probability of winning is 0.8 for each game. 
'''
n=10000000                # times of simulation
lst=[]
for i in range(n):
    match=1
    lose=0
    while match<=82:
        x=random.random()
        if x<=0.2:
            lose+=1      # calcuate the losing game
        else:
            lose==0     # if team win, the lose count reset to be 0.
        if lose==2:
            lst.append(1) # if continous lose happens, we mark that event as 1, and finish this round of simulation
            break
        lst.append(0)    # if continous lose didn't happen in whole regular season, then we mark that event as 0
        match+=1            # if continous lose didn;t happen until current game, the regular season contunues.
print(lst.count(1)/len(lst)) # calculate the sample probability of continuous lose. result is around 10%
        