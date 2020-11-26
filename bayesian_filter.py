# file created by Leonardo Cencetti on 11/23/20
import numpy as np
import pymc3 as pm


class BayesianFilter:
    def __init__(self, state_dim, output_dim):
        self.N = state_dim
        self.M = output_dim

    def compute_sigma_points(self):
        pass
