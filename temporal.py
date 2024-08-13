import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Parallel, delayed
import random
from tqdm import tqdm

class TemporalMatrixFactorizationWithNeighbors:
    def __init__(self, df_train, num_users, num_items, K, alpha, beta, iterations, early_stopping_rounds, df_val=None, df_propensity=None, neighborhood_size=5, user_col='userid', item_col='itemid', rating_col='rating', verbosity=0, n_jobs=-1, rating_min=1, rating_max=5, error_metric='rmse'):
        """
        Initialize the TemporalMatrixFactorizationWithNeighbors model.

        Parameters:
        df_train (pd.DataFrame): Training dataset.
        num_users (int): Number of users.
        num_items (int): Number of items.
        K (int): Number of latent factors.
        alpha (float): Learning rate.
        beta (float): Regularization parameter.
        iterations (int): Number of iterations.
        early_stopping_rounds (int): Early stopping rounds.
        df_val (pd.DataFrame, optional): Validation dataset. Defaults to None.
        df_propensity (pd.DataFrame, optional): Propensity scores dataset. Defaults to None.
        neighborhood_size (int, optional): Neighborhood size for similarity. Defaults to 5.
        user_col (str, optional): User column name. Defaults to 'userid'.
        item_col (str, optional): Item column name. Defaults to 'itemid'.
        rating_col (str, optional): Rating column name. Defaults to 'rating'.
        verbosity (int, optional): Verbosity level. Defaults to 0.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
        rating_min (int, optional): Minimum rating value. Defaults to 1.
        rating_max (int, optional): Maximum rating value. Defaults to 5.
        error_metric (str, optional): Error metric to optimize. Defaults to 'rmse'.
        """
        self.verbosity = verbosity
        self.df_train = self.preprocess(df_train, user_col, item_col, rating_col)
        self.df_val = self.preprocess(df_val, user_col, item_col, rating_col) if df_val is not None else None
        self.df_propensity = self.preprocess_propensity(df_propensity, user_col, item_col) if df_propensity is not None else None
        self.num_users = num_users
        self.num_items = num_items
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.neighborhood_size = neighborhood_size
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.n_jobs = n_jobs
        self.rating_min = rating_min
        self.rating_max = rating_max
        self.error_metric = error_metric

        if self.verbosity >= 2:
            print("Initializing propensity matrix")
        # Initialize propensity matrix
        if self.df_propensity is not None:
            self.propensity = self.df_propensity.pivot(index=self.user_col, columns=self.item_col, values='propensity').fillna(1).values
        else:
            self.propensity = np.ones((self.num_users, self.num_items))

    def preprocess(self, df, user_col, item_col, rating_col):
        """
        Preprocess the data by grouping and averaging ratings.

        Parameters:
        df (pd.DataFrame): DataFrame to preprocess.
        user_col (str): User column name.
        item_col (str): Item column name.
        rating_col (str): Rating column name.

        Returns:
        pd.DataFrame: Preprocessed DataFrame.
        """
        if self.verbosity >= 2:
            print("Preprocessing data")
        if df is not None:
            df = df.groupby([user_col, item_col], as_index=False)[rating_col].mean()
        return df

    def preprocess_propensity(self, df, user_col, item_col):
        """
        Preprocess the propensity data by grouping and averaging propensity scores.

        Parameters:
        df (pd.DataFrame): DataFrame to preprocess.
        user_col (str): User column name.
        item_col (str): Item column name.

        Returns:
        pd.DataFrame: Preprocessed propensity DataFrame.
        """
        if self.verbosity >= 2:
            print("Preprocessing propensity data")
        if df is not None:
            df = df.groupby([user_col, item_col], as_index=False)['propensity'].mean()
        return df

    def train(self):
        """
        Train the TemporalMatrixFactorizationWithNeighbors model.

        Returns:
        list: Training process with errors at each iteration.
        """
        if self.verbosity >= 2:
            print("Starting training")
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        self.b = np.mean(self.df_train[self.rating_col])

        self.samples = self.df_train[[self.user_col, self.item_col, self.rating_col]].values

        self.user_similarity = self.compute_similarity(self.df_train.pivot(index=self.user_col, columns=self.item_col, values=self.rating_col).fillna(0), 'user')
        self.item_similarity = self.compute_similarity(self.df_train.pivot(index=self.item_col, columns=self.user_col, values=self.rating_col).fillna(0), 'item')

        training_process = []
        best_error = float('inf')
        best_iter = -1
        best_P, best_Q = None, None
        best_b_u, best_b_i = None, None

        for i in range(self.iterations):
            if self.verbosity >= 2:
                print(f"Iteration {i+1} of {self.iterations}")
            np.random.shuffle(self.samples)
            self.sgd()
            train_error = self.compute_error(self.df_train)
            if self.df_val is not None:
                val_error = self.compute_error(self.df_val)
                training_process.append((i, train_error, val_error))
                if self.verbosity >= 1:
                    print(f"Iteration: {i+1} ; Train {self.error_metric.upper()} = {train_error:.4f} ; Validation {self.error_metric.upper()} = {val_error:.4f}")

                if val_error < best_error:
                    best_error = val_error
                    best_iter = i
                    best_P = self.P.copy()
                    best_Q = self.Q.copy()
                    best_b_u = self.b_u.copy()
                    best_b_i = self.b_i.copy()
                elif i - best_iter >= self.early_stopping_rounds:
                    if self.verbosity >= 1:
                        print(f"Early stopping at iteration {i+1}")
                    break
            else:
                training_process.append((i, train_error))
                if self.verbosity >= 1:
                    print(f"Iteration: {i+1} ; Train {self.error_metric.upper()} = {train_error:.4f}")

        if self.df_val is not None:
            self.P = best_P
            self.Q = best_Q
            self.b_u = best_b_u
            self.b_i = best_b_i

        if self.verbosity >= 2:
            print("Training complete")
        return training_process

    def compute_similarity(self, df, axis):
        """
        Compute the cosine similarity matrix.

        Parameters:
        df (pd.DataFrame): DataFrame to compute similarity from.
        axis (str): 'user' for user similarity, 'item' for item similarity.

        Returns:
        np.ndarray: Similarity matrix.
        """
        if self.verbosity >= 2:
            print(f"Computing {axis} similarity")
        
        sparse_matrix = csr_matrix(df.values if axis == 'user' else df.values.T)
        n = sparse_matrix.shape[0]

        def compute_chunk(start, end):
            if self.verbosity >= 2:
                print(f"Computing chunk from {start} to {end}")
            return cosine_similarity(sparse_matrix[start:end], sparse_matrix)

        chunk_size = max(1, n // (self.n_jobs * 2))  # Ensure chunk_size is at least 1
        results = Parallel(n_jobs=self.n_jobs, backend='threading')(delayed(compute_chunk)(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size))

        if not results:
            raise ValueError("No chunks were processed. Check chunking logic or data size.")

        similarity = np.vstack(results)

        return similarity

    def sgd(self):
        """
        Perform stochastic gradient descent (SGD) to update the model parameters.
        """
        if self.verbosity >= 2:
            print("Performing SGD")
        users = self.samples[:, 0].astype(int)
        items = self.samples[:, 1].astype(int)
        ratings = self.samples[:, 2]

        valid_indices = (users < self.num_users) & (items < self.num_items)

        predictions = self.predict(users[valid_indices], items[valid_indices])
        errors = (ratings[valid_indices] - predictions) * self.propensity[users[valid_indices], items[valid_indices]]

        self.b_u[users[valid_indices]] += self.alpha * (errors - self.beta * self.b_u[users[valid_indices]])
        self.b_i[items[valid_indices]] += self.alpha * (errors - self.beta * self.b_i[items[valid_indices]])

        for idx in range(len(users[valid_indices])):
            i, j = users[valid_indices][idx], items[valid_indices][idx]
            e = errors[idx]
            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i, :])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j, :])

    def predict(self, users, items):
        """
        Predict ratings for given users and items.

        Parameters:
        users (np.ndarray): Array of user IDs.
        items (np.ndarray): Array of item IDs.

        Returns:
        np.ndarray: Predicted ratings.
        """
        if self.verbosity >= 2:
            print("Predicting ratings")
        valid_indices = (users < self.num_users) & (items < self.num_items)

        mf_predictions = np.zeros(users.shape)
        mf_predictions[valid_indices] = (
            self.b
            + self.b_u[users[valid_indices]]
            + self.b_i[items[valid_indices]]
            + np.einsum('ij,ij->i', self.P[users[valid_indices]], self.Q[items[valid_indices]])
        )

        neighbor_predictions = np.zeros(users.shape)
        neighbor_predictions[valid_indices] = self.neighbor_adjustment(users[valid_indices], items[valid_indices])

        predictions = mf_predictions + neighbor_predictions

        # Scale the predictions to the specified range
        predictions = np.clip(predictions, self.rating_min, self.rating_max)

        return predictions

    def predict_single(self, user, item):
        """
        Predict the rating for a single user and item.

        Parameters:
        user (int): User ID.
        item (int): Item ID.

        Returns:
        float: Predicted rating.
        """
        return self.predict(np.array([user]), np.array([item]))[0]

    def neighbor_adjustment(self, users, items):
        """
        Compute neighbor adjustments for predictions.

        Parameters:
        users (np.ndarray): Array of user IDs.
        items (np.ndarray): Array of item IDs.

        Returns:
        np.ndarray: Neighbor adjustments.
        """
        if self.verbosity >= 2:
            print("Computing neighbor adjustments")
        user_adjustments = np.zeros(users.shape)
        item_adjustments = np.zeros(items.shape)

        for idx in range(len(users)):
            i, j = users[idx], items[idx]

            if i >= self.user_similarity.shape[0] or j >= self.item_similarity.shape[0]:
                continue

            user_neighbors = np.argsort(-self.user_similarity[int(i), :])[:self.neighborhood_size]
            item_neighbors = np.argsort(-self.item_similarity[int(j), :])[:self.neighborhood_size]

            user_ratings = self.df_train[(self.df_train[self.user_col].isin(user_neighbors)) & (self.df_train[self.item_col] == j)]
            item_ratings = self.df_train[(self.df_train[self.user_col] == i) & (self.df_train[self.item_col].isin(item_neighbors))]

            if not user_ratings.empty:
                user_adjustments[idx] = user_ratings[self.rating_col].mean()

            if not item_ratings.empty:
                item_adjustments[idx] = item_ratings[self.rating_col].mean()

        user_adjustments[np.isnan(user_adjustments)] = 0
        item_adjustments[np.isnan(item_adjustments)] = 0

        return (user_adjustments + item_adjustments) / 2

    def compute_error(self, df):
        """
        Compute the selected error metric.

        Parameters:
        df (pd.DataFrame): DataFrame with actual ratings.

        Returns:
        float: Computed error.
        """
        if self.error_metric == 'rmse':
            return self.rmse(df)
        elif self.error_metric == 'mae':
            return self.mae(df)
        elif self.error_metric == 'maoe':
            return self.maoe(df)
        elif self.error_metric == 'mape':
            return self.mape(df)
        else:
            raise ValueError(f"Unsupported error metric: {self.error_metric}")

    def rmse(self, df):
        """
        Compute Root Mean Squared Error (RMSE).

        Parameters:
        df (pd.DataFrame): DataFrame with actual ratings.

        Returns:
        float: RMSE value.
        """
        if self.verbosity >= 2:
            print("Computing RMSE")
        users = df[self.user_col].values.astype(int)
        items = df[self.item_col].values.astype(int)
        ratings = df[self.rating_col].values

        valid_indices = (users < self.num_users) & (items < self.num_items)

        predictions = self.predict(users[valid_indices], items[valid_indices])
        errors = (ratings[valid_indices] - predictions) ** 2

        return np.sqrt(np.mean(errors))

    def mae(self, df):
        """
        Compute Mean Absolute Error (MAE).

        Parameters:
        df (pd.DataFrame): DataFrame with actual ratings.

        Returns:
        float: MAE value.
        """
        if self.verbosity >= 2:
            print("Computing MAE")
        users = df[self.user_col].values.astype(int)
        items = df[self.item_col].values.astype(int)
        ratings = df[self.rating_col].values

        valid_indices = (users < self.num_users) & (items < self.num_items)

        predictions = self.predict(users[valid_indices], items[valid_indices])
        errors = np.abs(ratings[valid_indices] - predictions)

        return np.mean(errors)

    def maoe(self, df):
        """
        Compute Mean Absolute Overestimation Error (MAOE).

        Parameters:
        df (pd.DataFrame): DataFrame with actual ratings.

        Returns:
        float: MAOE value.
        """
        if self.verbosity >= 2:
            print("Computing MAOE")
        users = df[self.user_col].values.astype(int)
        items = df[self.item_col].values.astype(int)
        ratings = df[self.rating_col].values

        valid_indices = (users < self.num_users) & (items < self.num_items)

        predictions = self.predict(users[valid_indices], items[valid_indices])
        errors = np.abs(ratings[valid_indices] - predictions) / np.where(ratings[valid_indices] != 0, ratings[valid_indices], 1)

        return np.mean(errors)

    def mape(self, df):
        """
        Compute Mean Absolute Percentage Error (MAPE).

        Parameters:
        df (pd.DataFrame): DataFrame with actual ratings.

        Returns:
        float: MAPE value.
        """
        if self.verbosity >= 2:
            print("Computing MAPE")
        users = df[self.user_col].values.astype(int)
        items = df[self.item_col].values.astype(int)
        ratings = df[self.rating_col].values

        valid_indices = (users < self.num_users) & (items < self.num_items)

        predictions = self.predict(users[valid_indices], items[valid_indices])
        errors = np.abs((ratings[valid_indices] - predictions) / np.where(ratings[valid_indices] != 0, ratings[valid_indices], 1))

        return np.mean(errors)

    def full_matrix(self):
        """
        Compute the full predicted rating matrix.

        Returns:
        np.ndarray: Full predicted rating matrix.
        """
        if self.verbosity >= 2:
            print("Computing full matrix")
        user_indices = np.arange(self.num_users)
        item_indices = np.arange(self.num_items)

        full_pred = np.array([[self.predict(user, item) for item in item_indices] for user in user_indices])

        return full_pred
    
    def full_matrix_new(self):
        """
        Updated Compute the full predicted rating matrix.
    
        Returns:
        np.ndarray: Full predicted rating matrix.
        """
        if self.verbosity >= 2:
            print("Computing full matrix")
        
        def predict_user_chunk(start, end):
            user_indices = np.arange(start, end)
            item_indices = np.arange(self.num_items)
            user_predictions = np.zeros((end - start, self.num_items))
            
            for i, user in enumerate(user_indices):
                user_predictions[i, :] = self.predict(np.full(self.num_items, user), item_indices)
            
            return user_predictions
        
        chunk_size = max(1, self.num_users // (self.n_jobs * 2))  # Ensure chunk_size is at least 1
        results = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(predict_user_chunk)(i, min(i + chunk_size, self.num_users)) for i in range(0, self.num_users, chunk_size)
        )
        
        if not results:
            raise ValueError("No chunks were processed. Check chunking logic or data size.")
        
        full_pred = np.vstack(results)
        return full_pred

    def full_matrix_with_prog(self):
        """
        Compute the full predicted rating matrix with progress.
    
        Returns:
        np.ndarray: Full predicted rating matrix.
        """
        if self.verbosity >= 2:
            print("Computing full matrix")
        
        def predict_user_chunk(start, end):
            user_indices = np.arange(start, end)
            item_indices = np.arange(self.num_items)
            user_predictions = np.zeros((end - start, self.num_items))
            
            for i, user in enumerate(user_indices):
                user_predictions[i, :] = self.predict(np.full(self.num_items, user), item_indices)
            
            return user_predictions
        
        chunk_size = max(1, self.num_users // (self.n_jobs * 2))  # Ensure chunk_size is at least 1
        total_chunks = (self.num_users + chunk_size - 1) // chunk_size  # Total number of chunks
    
        results = []
        with tqdm(total=total_chunks, desc="Computing full matrix") as pbar:
            parallel = Parallel(n_jobs=self.n_jobs, backend='threading')
            tasks = (delayed(predict_user_chunk)(i, min(i + chunk_size, self.num_users)) for i in range(0, self.num_users, chunk_size))
            results = parallel(tasks)
            pbar.update(total_chunks)  # Update the progress bar by the total number of chunks
    
        if not results:
            raise ValueError("No chunks were processed. Check chunking logic or data size.")
        
        full_pred = np.vstack(results)
        return full_pred

    
    def predict_all(self, num_iter, df, err_met, tmfs):
        """
        Predict ratings for random user-item combinations for all specified error metrics.

        Parameters:
        num_iter (int): Number of iterations to run.
        df (pd.DataFrame): DataFrame with user-item combinations.
        err_met (list): List of error metrics.
        tmfs (list): List of trained models.
        """
        unique_user_ids = df[self.user_col].unique()
        unique_book_ids = df[self.item_col].unique()
        for iter in range(num_iter):
            uid, bid = None, None
            
            # Loop until a valid combination is found
            while True:
                uid = random.choice(unique_user_ids)
                bid = random.choice(unique_book_ids)
                # Check if the combination exists and has a rating
                if not df[(df[self.user_col] == uid) & (df[self.item_col] == bid) & df[self.rating_col].notna()].empty:
                    break
            
            print("Randomly selected valid combination:")
            print(f"{self.user_col}:", uid)
            print(f"{self.item_col}:", bid)
            
            filtered_df = df[(df[self.user_col] == uid) & (df[self.item_col] == bid)]
            print(filtered_df)
            
            for err in err_met:
                for utmf in tmfs:
                    if utmf.error_metric == err:
                        pred = utmf.predict_single(uid, bid)
                        print(f'For error metric {err}, prediction is {pred}')
            print('-------------------------------')

    pass 
