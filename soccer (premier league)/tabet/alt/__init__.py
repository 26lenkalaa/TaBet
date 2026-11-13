"""Alternative team-strength models, each compared with the frozen Dixon-Coles model (V0).

returning_reset   option 2: a team that missed the previous PL season starts a fresh rating
bayes_dc          option 1: hierarchical (approximate) Bayesian Dixon-Coles with uncertainty averaging
goal_distributions option 4: negative binomial and bivariate Poisson goal distributions
(option 3, the Elo blend, lives in tabet/elo.py.)
"""
