# Sakai-and-Sakai-et-al._CellReports_2022

Parameter estimation code for Sakai-and-Sakai-et-al._CellReports_2022 using hyperopt 0.1.2.

Please see our paper for details (Sakai, Y., Sakai, Y. (co-first author), Abe, Y., Narumoto, J., & Tanaka, S. C. (2022). Memory trace imbalance in reinforcement and punishment systems can reinforce implicit choices leading to obsessive-compulsive behavior. Cell Reports, 40(9). https://doi.org/10.1016/j.celrep.2022.111275 ).

Please see https://github.com/YukiSakai1209/Sakai-and-Sakai-et-al._CellReports_2022 for the latest version.

# Brief description of codes
code_01_para_est.py: Estimate parameters using 100 different random seeds.

code_02_choose_optimal_para.py: Extract estimated parameters as the ones with the minimum negative log likelihood among 100 estimations with different random seeds.
