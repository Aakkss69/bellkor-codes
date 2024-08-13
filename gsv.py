#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:45:12 2024

@author: ayaks69
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from temporal import TemporalMatrixFactorizationWithNeighbors
from temporal_tuner import HyperparameterTuner

class CrossValidator:
    def __init__(self, model_class, param_grid, df, num_users, num_items, user_col='userid', item_col='itemid', rating_col='rating', n_splits=5, n_jobs=-1, verbosity=0):
        """
        Initialize the CrossValidator.

        Parameters:
        model_class (class): The model class to tune.
        param_grid (dict): Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values.
        df (pd.DataFrame): Dataset for cross-validation.
        num_users (int): Number of users.
        num_items (int): Number of items.
        user_col (str, optional): User column name. Defaults to 'userid'.
        item_col (str, optional): Item column name. Defaults to 'itemid'.
        rating_col (str, optional): Rating column name. Defaults to 'rating'.
        n_splits (int, optional): Number of splits for cross-validation. Defaults to 5.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
        verbosity (int, optional): Verbosity level. Defaults to 0.
        """
        self.model_class = model_class
        self.param_grid = param_grid
        self.df = df
        self.num_users = num_users
        self.num_items = num_items
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.n_splits = n_splits
        self.n_jobs = n_jobs
        self.verbosity = verbosity
        self.cv_results = []

    def cross_validate(self):
        """
        Perform cross-validation and return the average score for each parameter combination.

        Returns:
        list: Cross-validation results.
        """
        kf = KFold(n_splits=self.n_splits)
        fold = 1

        for train_index, val_index in kf.split(self.df):
            if self.verbosity >= 1:
                print(f"Processing fold {fold}/{self.n_splits}")

            df_train, df_val = self.df.iloc[train_index], self.df.iloc[val_index]

            tuner = HyperparameterTuner(
                model_class=self.model_class,
                param_grid=self.param_grid,
                num_users=self.num_users,
                num_items=self.num_items,
                df_train=df_train,
                df_val=df_val,
                user_col=self.user_col,
                item_col=self.item_col,
                rating_col=self.rating_col,
                n_jobs=self.n_jobs,
                verbosity=self.verbosity
            )

            best_result = tuner.fit()
            self.cv_results.append(best_result)

            if self.verbosity >= 1:
                print(f"Best parameters for fold {fold}: {best_result['best_params']}")
                print(f"Best score for fold {fold}: {best_result['best_score']}")
                print('------------------------------------------------------')
                print('\n\n\n\n')
            
            fold += 1

        return self.cv_results

    def get_average_best_score(self):
        """
        Calculate the average best score from cross-validation results.

        Returns:
        float: Average best score.
        """
        total_score = sum(result['best_score'] for result in self.cv_results)
        return total_score / len(self.cv_results)

    def get_best_params(self):
        """
        Get the best parameters from cross-validation results.

        Returns:
        dict: Best parameters.
        """
        best_params_count = {}
        for result in self.cv_results:
            params = tuple(result['best_params'].items())
            if params not in best_params_count:
                best_params_count[params] = 0
            best_params_count[params] += 1
        
        best_params = max(best_params_count, key=best_params_count.get)
        return dict(best_params)