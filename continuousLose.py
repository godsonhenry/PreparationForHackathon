# -*- coding: utf-8 -*-
import random

'''
Calculate the probability of continuous lose of a team in a regular season. The probability of winning is 0.8 for each game. 
'''
n=10000000                # times of simulation
count=0                   # count the possible season in which continous lose happens.
for i in range(n):
    match=1
    lose=0
    while match<=82:
        x=random.random()
        if x<=0.2:
            lose+=1      # calcuate the losing game
        else:
            lose=0     # if team win, the lose count reset to be 0.
        if lose==2:
            count+=1 # if continous lose happens, count increase 1, and finish this round of simulation
            break         
        match+=1            # if continous lose didn;t happen until current game, the regular season contunues.
print(count/n )# calculate the sample probability of continuous lose. result is around 94%
