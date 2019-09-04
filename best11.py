# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 23:28:16 2018

@author: Adarsh kush
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

dbs=pd.read_csv("batsmandata.csv")
dblr=pd.read_csv("bowlerdata.csv")
dwk=pd.read_csv("wkeeperdata.csv")
dalr=pd.read_csv("allrounderdata.csv")
del dbs['ind']
del dblr['ind']
#we are going with combination of 5 batsman 2 allrounders and 4 bowlers
#let's choose bowler first
#assign weights
a=0.5
b=1
c=2
d=3
dbs['Batsmans']=(d*dbs.Avg+c*dbs.SR+a*dbs.Avg_fours+a*dbs.Avg_sixes+b*dbs.Thirty_Plus+c*dbs.Notouts)/(2*a+1*b+1*c+2*d)
plt.figure(figsize=(15,6))
dbs1=dbs.sort_values('Batsmans',ascending=False)[:5]
x1=np.array(list(dbs1['Name']))
y1=np.array(list(dbs1['Batsmans']))

sn.barplot(x1,y1,palette="colorblind")
plt.ylabel("Batsmans rating")
#lets choose alrounders
dalr['Allrounders']=(c*dalr.Runs+d*dalr.Ave+b*dalr.Hundreds+c*dalr.Wkts+d*dalr.Bave+b*dalr.Fifer+b*dalr.Ct)/(0*a+3*b+2*c+2*d)
plt.figure(figsize=(15,3))
dalr1=dalr.sort_values('Allrounders',ascending=False)[:2]
x2=np.array(list(dalr1['Player']))
y2=np.array(list(dalr1['Allrounders']))

sn.barplot(x2,y2,palette="colorblind")
plt.ylabel("Allrounders rating")

#lets choose wicket keeper
dwk['Wicketkeeper']=(c*dwk.Dis+c*dwk.St+c*dwk.Ct+d*dwk.dispinn)/(3*c+1*d)
plt.figure(figsize=(8,8))
dwk1=dwk.sort_values('Wicketkeeper',ascending=False)[:1]
x3=np.array(list(dwk1['Player']))
y3=np.array(list(dwk1['Wicketkeeper']))

sn.barplot(x3,y3,palette="colorblind")
plt.ylabel("Best wicketkeeper")

#lets choose bowlers
dblr['Bowlers']=(d*dblr.bowl_avg+c*dblr.bowl_sr+b*dblr.bowl_dots+b*dblr.two_pluswkts)/(2*b+1*c+1*d)
plt.figure(figsize=(15,6))
dblr1=dblr.sort_values('Bowlers',ascending=False)[:4]
x4=np.array(list(dblr1['Name']))
y4=np.array(list(dblr1['Bowlers']))

sn.barplot(x4,y4,palette="bright")
plt.ylabel("Bowlers rating")