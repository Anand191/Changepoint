#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:49:57 2017

@author: anand
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%
df = pd.read_csv("arnhem.csv",sep=";");
df['CPU_TOTAL_SECONDS'] = df['CPU_TOTAL_SECONDS'].apply(lambda x:x.replace(',','.'))
df['CPU_TOTAL_SECONDS'] = pd.to_numeric(df['CPU_TOTAL_SECONDS'] )
df['MSU_CPU'] = pd.to_numeric(df['MSU_CPU'] )
df['MSU_CPU'] = df['MSU_CPU'].apply(lambda x:x.replace(',','.'))
df['MEASURED_SEC'] = df['MEASURED_SEC'].apply(lambda x:x.replace(',','.'))
df['TIME'] = df['TIME'].apply(lambda x: x.split('.')[0])
df['TIMESTAMP'] = df['FK_DateKey'].astype(str) + " " + df['TIME'].astype(str)
df.iloc[:,-1] = pd.to_datetime(df.iloc[:,-1])
df['JOB_NAME'] = df['JOB_NAME'].str.strip()
df = df.sort_values(by=['TIMESTAMP'])

jobs = df['JOB_NAME'].unique().tolist()

#%%
z = 1
plt.figure(figsize=(14,10))
for job in jobs:
    rows = np.where(df['JOB_NAME']==job)[0]
    X = df['TIMESTAMP'].iloc[rows]
    Y = df['CPU_TOTAL_SECONDS'].iloc[rows]
    plt.subplot(5,2,z)
    plt.plot(X,Y)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.title("%s"%job)
    z+=1