# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 14:11:16 2018

@author: Yuki Sakai
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

# parameters
random_seeds = np.arange(1,101)
variables = ['alpha_est', 'beta_est', 'nu_posi_est', 'nu_nega_est', 'nu_posi_est-nu_nega_est', 'nll']
# subjects
demo = pd.read_csv('demo.csv')
subs = list(demo['Sub_ID'])
subs.sort()

"""
Extract estimated parameters as the ones with the minimum negative log likelihood among 100 estimations with different random seeds.
"""
for s in tqdm(subs):
    # initialize dict
    para_est = {}
    for v in variables:
        para_est[v] = []
    # collect estimated paras
    for i in [98, 99, 100]:
        tmp = pd.read_csv('para_est_{}.csv'.format(i))
        for v in variables:
            para_est[v].append(tmp.loc[tmp[tmp['Sub_ID']==s].index, v].values[0])
    # choose optimal
    for v in variables:
        demo.loc[demo[demo['Sub_ID']==s].index, v] = para_est[v][np.argmin(para_est['nll'])]
demo.to_csv('demo_para_est.csv')
