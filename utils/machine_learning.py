from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import pandas as pd


class Rounder(BaseEstimator, TransformerMixin):
    def __init__(self, decimals=0):
        self.decimals = decimals

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.round(X, decimals=self.decimals)
