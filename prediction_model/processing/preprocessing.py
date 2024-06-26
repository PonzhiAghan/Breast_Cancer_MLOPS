from sklearn.base import BaseEstimator,TransformerMixin
from prediction_model.config import config
import numpy as np

class MeanImputer (BaseEstimator,TransformerMixin):
    
    def __init__(self, variables=None):
        self.variables = variables
        
    def fit(self,X,y=None):
        self.mean_dict = {}
        for col in self.variables:
          self.mean_dict[col] = X[col].mean()
        return self
    
    def transform(self,X):
       X=X.copy()
       for col in self.variables:
          X[col].fillna(self.mean_dict[col], inplace=True)
       return X

class ModeImputer(BaseEstimator,TransformerMixin):
    def __init__(self, variables=None):
       self.variables = variables

    def fit(self,X,y=None):
       self.mode_dict = {}
       for col in self.variables:
          self.mode_dict[col] = X[col].mode()[0]
       return self
    
    def transform(self,X):
       X=X.copy()
       for col in self.variables:
          X[col].fillna(self.mode_dict[col], inplace=True)
       return X
