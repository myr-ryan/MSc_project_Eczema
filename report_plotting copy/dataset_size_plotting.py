import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SWET_DIR = "SWET_files.csv"
TLA_DIR = "TLA_files.csv"

SWET_files = pd.read_csv(SWET_DIR).values.tolist()
TLA_files = pd.read_csv(TLA_DIR).values.tolist()


for i in range(len(SWET_files)):
    tup = SWET_files[i][0].replace('(', '')
    tup = tup.replace(')', '')
    tup = tuple(map(int, tup.split(',')))
    x = tup[0]
    y = tup[1]
    size = int(SWET_files[i][1])
    # print(size)

    # print(tuple(map(int, tup.split(','))))

    plt.scatter(x, y, s=size, color='#1f77b4')

for i in range(len(TLA_files)):
    tup = TLA_files[i][0].replace('(', '')
    tup = tup.replace(')', '')
    tup = tuple(map(int, tup.split(',')))
    x = tup[0]
    y = tup[1]
    size = int(TLA_files[i][1])

    # print(tuple(map(int, tup.split(','))))

    plt.scatter(x, y, s=size, color='#d62728')

plt.rcParams['font.size'] = '18'
plt.xlabel('Height', fontsize=18)
plt.ylabel('Width', fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
blue_patch = mpatches.Patch(color='#1f77b4', label='SWET')
red_patch = mpatches.Patch(color='#d62728', label='TLA')
plt.legend(handles=[blue_patch, red_patch])



plt.show()
