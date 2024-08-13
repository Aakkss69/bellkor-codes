#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 13:20:55 2024

@author: ayaks69
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

class HyperparameterTuner:
    def __init__(self, model_class, param_grid, num_users, num_items, df_train, df_val=None, df_propensity=None, user_col='userid', item_col='itemid', rating_col='rating', n_jobs=-1, verbosity=0):
        """
        Initialize the HyperparameterTuner.

        Parameters:
        model_class (class): The model class to tune.
        param_grid (dict): Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values.
        num_users (int): Number of users.
        num_items (int): Number of items.
        df_train (pd.DataFrame): Training dataset.
        df_val (pd.DataFrame, optional): Validation dataset. Defaults to None.
        df_propensity (pd.DataFrame, optional): Propensity scores dataset. Defaults to None.
        user_col (str, optional): User column name. Defaults to 'userid'.
        item_col (str, optional): Item column name. Defaults to 'itemid'.
        rating_col (str, optional): Rating column name. Defaults to 'rating'.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
        verbosity (int, optional): Verbosity level. Defaults to 0.
        """
        self.model_class = model_class
        self.param_grid = list(ParameterGrid(param_grid))
        self.num_users = num_users
        self.num_items = num_items
        self.df_train = df_train
        self.df_val = df_val
        self.df_propensity = df_propensity
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.best_params = None
        self.best_score = float('inf')
        self.best_model = None

    def fit(self):
        """
        Fit the hyperparameter tuner.

        Returns:
        dict: Best parameters and corresponding score.
        """
        for i, params in enumerate(self.param_grid):
            if self.verbosity >= 1:
                print(f"Training model {i+1}/{len(self.param_grid)} with parameters: {params}")

            model = self.model_class(
                df_train=self.df_train,
                num_users=self.num_users,
                num_items=self.num_items,
                df_val=self.df_val,
                df_propensity=self.df_propensity,
                user_col=self.user_col,
                item_col=self.item_col,
                rating_col=self.rating_col,
                n_jobs=self.n_jobs,
                verbosity=self.verbosity,
                **params
            )

            model.train()

            train_error = model.compute_error(self.df_train)
            if self.df_val is not None:
                val_error = model.compute_error(self.df_val)
                combined_error = (train_error + val_error) / 2
                if self.verbosity >= 1:
                    print(f"Train {model.error_metric.upper()} for model {i+1}/{len(self.param_grid)}: {train_error}")
                    print(f"Validation {model.error_metric.upper()} for model {i+1}/{len(self.param_grid)}: {val_error}")
                    print(f"Combined {model.error_metric.upper()} for model {i+1}/{len(self.param_grid)}: {combined_error}")

                if combined_error < self.best_score:
                    self.best_score = combined_error
                    self.best_params = params
                    self.best_model = model
            else:
                if self.verbosity >= 1:
                    print(f"Train {model.error_metric.upper()} for model {i+1}/{len(self.param_grid)}: {train_error}")

                if train_error < self.best_score:
                    self.best_score = train_error
                    self.best_params = params
                    self.best_model = model

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_model': self.best_model
        }

    def predict(self, user, item):
        """
        Predict the rating for a single user and item using the best model.

        Parameters:
        user (int): User ID.
        item (int): Item ID.

        Returns:
        float: Predicted rating.
        """
        if self.best_model is None:
            raise ValueError("No best model found. Please run the fit method first.")
        
        return self.best_model.predict_single(user, item)
    
    def get_best_model(self):
        return self.best_model
    pass