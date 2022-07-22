# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 06:36:51 2018

@author: Yuki Sakai
"""
import os
import numpy as np
import pandas as pd
from hyperopt import hp, tpe, Trials, fmin
from joblib import Parallel, delayed
from scipy.stats import gamma as gamma_distribution

"""
parameters
"""
gamma = 0
max_evals = 50000
n_jobs = -1
max_beta = 100
max_others = 0.95
random_seeds = np.arange(1,101)
exp_file = 'data.csv'


"""
maximum a posteriori estimation
"""

class ML(object):
    def __init__(self, df, seed):
        # experimental settings
        self.n_states = 16
        self.possible_actions = np.array([[0, 2],
                                        [1, 3],
                                        [0, 1],
                                        [2, 3],
                                        [1, 2],
                                        [0, 3],
                                        [4, 6],
                                        [5, 7],
                                        [5, 4],
                                        [7, 6],
                                        [4, 7],
                                        [5, 6],
                                        [2, 4],
                                        [0, 6],
                                        [3, 5],
                                        [1, 7]
                                         ])
        self.rewards = np.array([40, 40, 10, 10, -10, -10, -40, -40])
        self.delays =  np.array([0,  3,  0,  3,  0,   3,   0,   3])

        self.df = df
        self.seed = seed

    def softmax(self, a, beta):
        a = beta * a
        p = np.exp(a-np.max(a)) / np.sum(np.exp(a-np.max(a)))
        p[np.where(p==1)] = 1-np.finfo(np.double).tiny
        p[np.where(p==0)] = np.finfo(np.double).tiny
        return p
    
    def neg_ll_actor_critic(self, args):
        """
        actor-critic
        """
        alpha, beta, nu_posi, nu_nega = args
        df = self.df

        # initialize log likelihood
        ll = 0
        # load data
        state, action, reward, dir_stim_or_error = df['S'].values.astype(
            int), df['A'].values.astype(int), df['R'].values.astype(int), df['Dir_stim_or_error'].values.astype(int)
        # -1 only for python index start from 0
        state -= 1
        action -= 1
        # create matrix
        w = np.zeros((self.n_states, 2), dtype=np.float64) # policy parameters
        e_p = np.zeros((self.n_states, 2), dtype=np.float64) # eligibility traces for positive prediction error
        e_n = np.zeros((self.n_states, 2), dtype=np.float64) # eligibility traces for negative prediction error

        for t, (s, a, r, error) in enumerate(zip(state, action, reward, dir_stim_or_error)):
            if error == -1: # button press error: -1
                # only decay eligibility traces in the case of button press error.
                if t < df.shape[0] - 1:
                    e_p *= nu_posi
                    e_n *= nu_nega
            else:
                if a == self.possible_actions[s, 0]:
                    action_idx = 0
                else:
                    action_idx = 1
                # likelihood
                p = self.softmax(w[s, :], beta)
                # log likelihood of selected action
                ll += np.log(p[action_idx])
                # update
                if t < df.shape[0] - 1:
                    prediction_error = r + gamma * w[state[t+1], :].sum() - w[s, :].sum()
                    # eligibility traces for positive prediction error
                    e_p *= nu_posi
                    e_p[s, action_idx] += 1
                    # eligibility traces for negative prediction error
                    e_n *= nu_nega
                    e_n[s, action_idx] += 1
                    # w
                    if prediction_error >= 0:
                        w += alpha * prediction_error * e_p
                    elif prediction_error < 0:
                        w += alpha * prediction_error * e_n
        # beta: gamma(2, 3) prior
        ll += np.log( gamma_distribution.pdf(beta,2,scale=3) )
        
        # return negative log likelihood
        return -ll

    def ml_estimation(self):
        """
        SMBO
        """
        seed = self.seed
        hyperopt_parameters = [
            hp.uniform('alpha', 0, max_others),
            hp.uniform('beta', 0, max_beta),
            hp.uniform('nu_posi', 0, max_others),
            hp.uniform('nu_nega', 0, max_others)
        ]
        trials = Trials()
        best = fmin(
            # objective function to be minimized
            self.neg_ll_actor_critic,
            # parameters to be estimated
            hyperopt_parameters,
            # logic
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate= np.random.RandomState(seed),
            verbose=1
        )
        return best, trials
    
def wrap(i, df_s, seed):
    res = np.zeros((1, 9))
    ml = ML(df_s, seed)
    res_tmp, trials = ml.ml_estimation()
    res[0, 0:4] = np.array([res_tmp['alpha'], res_tmp['beta'], res_tmp['nu_posi'], res_tmp['nu_nega']])
    res[0, 4] = res_tmp['nu_posi'] - res_tmp['nu_nega']
    res[0, 5] = trials.best_trial['result']['loss']
    res[0, 6] = df_s['GID'].unique()
    res[0, 7] = df_s['GID_med_dose'].unique()
    res[0, 8] = df_s['Sub_num'].unique()
    print('{}th subject finished'.format(i))
    return res

def fit_behavioral_data(data, seed):
    """Fit a model for all subs. """
    df = pd.read_csv(data, index_col=0)
    subjects = df['Sub_num'].unique()
    res = Parallel(n_jobs=n_jobs)([delayed(wrap)(i+1, df[df['Sub_num'] == i+1], seed) for i in range(len(subjects))])
    tmp = np.empty((len(subjects), 9))
    for i in range(len(subjects)):
        tmp[i] = res[i]
    res_merged = pd.DataFrame(tmp, pd.Index(subjects, name='sub'),
                              ['alpha_est', 'beta_est', 'nu_posi_est', 'nu_nega_est', 'nu_posi_est-nu_nega_est',
                               'nll', 'GID', 'GID_med_dose', 'Sub_num'])
    res_merged = res_merged.sort_values(by=['Sub_num'], ascending=True)
    ID = []
    for i in range(len(subjects)):
        ID.append(df[df['Sub_num'] == i+1]['Sub_ID'].unique()[0])
    res_merged['Sub_ID'] = ID
    return res_merged

        
if __name__ == '__main__':
    data = exp_file
    # Estimate parameters using 100 different random seeds
    for seed in random_seeds:
        if not os.path.exists('para_est_' + str(seed) + '.csv'): # only for parallel calc on different pc.
            tmp = pd.DataFrame()
            tmp.to_csv('para_est_' + str(seed) + '.csv')
            res_merged = fit_behavioral_data(data, seed)
            res_merged.to_csv('para_est_' + str(seed) + '.csv')
