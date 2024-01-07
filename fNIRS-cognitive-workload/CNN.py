
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sktime.classification.deep_learning.cnn import CNNClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
# from tensorflow.keras.callbacks import EarlyStopping

import pandas as pd
try:
    import importlib
    importlib.reload(pp)
except NameError: # It hasn't been imported yet
    import pre_processing as pp


class baseline_CNN:
    
    def __init__(self, data, split_label, drop_label=None, subject_by_subject=False, test_indexes=None, sub=None):

        ## If we are going to fit the data to all subjects combined ##
        if subject_by_subject == False:

            ## First group data by abel block ##
            all_grouped_dataframes, all_block_change_idx = pp.detect_block_changes(data, split_label)

            ## Create nested data ##
            X_dict = {}
            for task, task_df in all_grouped_dataframes.items():
                task_dict = {}
                sub = task_df.index.get_level_values(1).unique()[0] if split_label == 'chunk' else task_df.index.get_level_values(0).unique()[0]
                for col_name, col_data in task_df.items():
                    if col_name != 'difficulty':
                        task_dict[col_name] = pd.Series(col_data.values)
                X_dict[f'{task}_{sub}'] = task_dict
            X = pd.DataFrame(X_dict).T
            X = X if drop_label is None else X.drop(drop_label, axis=1)
            y = data[all_block_change_idx].difficulty

            ## Add relevant data to keep as attributes of class ##
            self.X_dict = X_dict
            self.X = X
            self.y = y
        else:
            ## Split subject data by sub-sample "chunks" ##
            sub_grouped_dataframes, sub_block_change_idx = pp.detect_block_changes(data, split_label)

            ## Find the range of test indexes for subject data, by referencing "test_indexes" ##
            test_idx = [idx for name, idx in test_indexes.items() if f'{sub}_' in name]
            test_ranges = [[x, x + 416] for x in test_idx]

            ## Create nested data ##
            train_dict = {}
            test_dict = {}
            for task, task_df in sub_grouped_dataframes.items():
                overall_block = task_df.index.get_level_values(0).unique()[0]
                if any(any(value in task_df.index.get_level_values(2) for value in range(start, end + 1)) for start, end in test_ranges):
                    task_test = {}
                    for col_name, col_data in task_df.items():
                        task_test[col_name] = pd.Series(col_data.values)
                    test_dict[f'{overall_block}_{task}_{sub}'] = task_test
                else:
                    task_train = {}
                    for col_name, col_data in task_df.items():
                        task_train[col_name] = pd.Series(col_data.values)
                    train_dict[f'{overall_block}_{task}_{sub}'] = task_train
            train = pd.DataFrame(train_dict).T
            test = pd.DataFrame(test_dict).T

            X_train, X_test = train.drop(['chunk', 'difficulty'], axis=1), test.drop(['chunk', 'difficulty'], axis=1)
            y_train, y_test = pd.Series([x.unique()[0] for x in train.difficulty], index=train.index), pd.Series([x.unique()[0] for x in test.difficulty], index=test.index)

            ## Add relevant data to keep as attributes of class ##
            self.X_train = X_train.sample(frac=1, random_state=42)
            self.X_test = X_test.sample(frac=1, random_state=42)
            self.y_train = y_train.sample(frac=1, random_state=42)
            self.y_test = y_test.sample(frac=1, random_state=42)
            self.test_indexes = test_indexes

    def split_data(self, test_size=0.2, train_sub=None, test_sub=None):
        if (train_sub is None) & (test_sub is None):
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        else:
            train_idx = [x for x in self.X.index.get_level_values(0) if ('_').join(x.split('_')[1:]) in train_sub]
            test_idx = [x for x in self.X.index.get_level_values(0) if ('_').join(x.split('_')[1:]) in test_sub]

            X_train, X_test = self.X[self.X.index.get_level_values(0).isin(train_idx)], self.X[self.X.index.get_level_values(0).isin(test_idx)]
            y_train, y_test = self.y[self.y.index.get_level_values(0).isin(train_sub)], self.y[self.y.index.get_level_values(0).isin(test_sub)]

        self.X_train = X_train.sample(frac=1, random_state=42)
        self.X_test = X_test.sample(frac=1, random_state=42)
        self.y_train = y_train.sample(frac=1, random_state=42)
        self.y_test = y_test.sample(frac=1, random_state=42)

    def train(self, epochs, cv=5):
        network = CNNClassifier(n_epochs=epochs, verbose=0, random_state=42)
        ypred_val = network.fit_predict(self.X_train, self.y_train, cv=cv, change_state=True)
        test_score = network.score(self.X_test, self.y_test)
        print('validation accuracy:', network.summary()['accuracy'][-1])
        print(f'test accuracy: {test_score}')
        
        self.trained_network = network
        self.test_score = test_score
        self.history = network.summary()
        self.test_predictions = network.predict(self.X_test)
        self.ypred_val = ypred_val

    def train_grid(self, epochs, param_grid = {"kernel_size": [7, 9], "avg_pool_size": [3, 5]}, n_iter=10):
        early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        network = CNNClassifier(n_epochs=epochs, verbose=1, random_state=42, callbacks=[early_stopping])
        grid = RandomizedSearchCV(network, param_distributions=param_grid, cv=5, random_state=42, n_iter=n_iter)
        grid.fit(self.X_train, self.y_train)

        print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
        print("Test set score: {:.2f}".format(grid.score(self.X_test, self.y_test)))
        print("Best parameters: {}".format(grid.best_params_))

        self.grid_results = grid