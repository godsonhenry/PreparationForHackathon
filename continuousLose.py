# -*- coding: utf-8 -*-
import random
from itertools import combinations as cb
import matplotlib.pyplot as plt

'''
Calculate the probability of continuous lose of a team in a regular season. The probability of winning is 0.8 for each game.
'''

def NBALose(n,p,season):
    count=0                   # count the possible season in which continous lose happens.
    for i in range(n):
        match=1
        lose=0
        while match<=season:
            x=random.random()
            if x<=1-p:
                lose+=1         # calcuate the losing game
            else:
                lose=0          # if team win, the lose count reset to be 0.
            if lose==2:
                count+=1        # if continous lose happens, count increase 1, and finish this round of simulation
                break
            match+=1            # if continous lose didn;t happen until current game, the regular season contunues.
    return count/n              # calculate the sample probability of continuous lose. result is around 94%

print('The probability of continuous lose of Warriors is', NBALose(100000,0.8,82))

# research the curve of win probability vs probability to find the turning point
# Figure (1) in docx file
p=0.1
x=[]
y=[]
while p<=1:
    x.append(p)
    y.append(1-NBALose(100000,p,82))
    p+=0.01

plt.plot(x,y)
plt.title('win probability Vs probability of no continuous lose')
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('win probability')
plt.ylabel('probability of no continuous lose')
plt.show()

# the curve increase emergently between 0.8 to 1.0 of p value

# further investigation between 0.8 to 1 of win probability
# Figure (2) in docx file
p=0.8
x=[]
y=[]
while p<=1:
    x.append(p)
    y.append(1-NBALose(100000,p,82))
    p+=0.01

plt.plot(x,y)
plt.plot(x,[0.5]*len(y),color='red')
plt.title('win probability Vs probability of no continuous lose')
plt.xlim(0.8,1)
plt.ylim(0,1)
plt.xlabel('win probability')
plt.ylabel('probability of no continuous lose')
plt.show()

# the proposed range is set from 0.875 to 0.925 of win probability
p=0.875
x=[]
y=[]
while p<=0.925:
    if NBALose(100000,p,82)>0.5:
        p+=0.001
        print(1-NBALose(1000,p,82),'under',p)
    else:
        print(1-NBALose(1000,p,82),'under',p)  #probability of win should exceed 90%
        break
