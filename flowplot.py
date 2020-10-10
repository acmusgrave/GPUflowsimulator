# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:38:25 2020

@author: Andrew
"""

import matplotlib.pyplot as plt
import pandas as pd

dataFrame = pd.read_csv("out_0_ms.csv")

position = dataFrame['X']
flowrate = dataFrame['Q']
area = dataFrame['A']

fig, [ax1, ax2] = plt.subplots(2, 1)

ax1.plot(position, flowrate, label='t = 0s')
ax2.plot(position, area)

dataFrame = pd.read_csv("out_709_ms.csv")

position = dataFrame['X']
flowrate = dataFrame['Q']
area = dataFrame['A']

ax1.plot(position, flowrate, label='t = 0.7s')
ax2.plot(position, area)

dataFrame = pd.read_csv("out_1410_ms.csv")

position = dataFrame['X']
flowrate = dataFrame['Q']
area = dataFrame['A']

ax1.plot(position, flowrate, label='t = 1.4s')
ax2.plot(position, area)

dataFrame = pd.read_csv("out_2111_ms.csv")

position = dataFrame['X']
flowrate = dataFrame['Q']
area = dataFrame['A']

ax1.plot(position, flowrate, label='t = 2.1s')
ax2.plot(position, area)


ax2.set_xlabel("x")
ax1.set_ylabel("Q")
ax2.set_ylabel("A")
ax1.legend(loc='upper left')
#ax1.set_xlim(0, 20)
#ax1.set_ylim(1, 3)
#ax2.set_xlim(0, 20)
#ax2.set_ylim(0, 3)


plt.show()