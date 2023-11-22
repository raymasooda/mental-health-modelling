#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import linregress
from scipy.stats import zscore
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def calculate_adjusted_R2(r2, n, k):
    adjusted_R2 = 1 - (((1-r2)) * (n-1) / (n-k-1))
                       
    return adjusted_R2


def generate_score(X, y, m):
    if m == 'accuracy':
        score = metrics.accuracy_score(X, y)
    elif m == 'recall':
        score = metrics.recall_score(X, y)
    elif m == 'precision':
        score = metrics.precision_score(X, y)

    return score
# In[1]:


def plot_bestfit_line(x,y,ax_obj):
    slope = linregress(x,y).slope #calculate slope of line
    intercept = linregress(x,y).intercept #calculate intercept of line
    
    xvalues = np.linspace(np.min(x), np.max(x), 100) #create an array of length 100 between the minimum and maximum x-values
    yvalues = xvalues*slope + intercept #evaluate the y-values at each of the x-values defined above
    ax_obj.plot(xvalues, yvalues, color='black', lw=3) #plot the line of x-values vs. y values to specified axes object


# In[2]:


def train_validate(model_type, X_train, X_test, y_train, y_test, nfeatures, nsamples, name, penalty=None, kernel=None, gamma=None, c=1, neighbors=5):
    ## Define new model ##
    if type(model_type) == str:
        if model_type == 'lin_reg':
            model = LinearRegression()
        if model_type == 'log_reg':
                model = LogisticRegression(penalty=penalty, solver='saga', C=c)
        if model_type == 'naive_bayes':
            if penalty is None:
                model = GaussianNB()
            else:
                model = GaussianNB(penalty=penalty, solver='saga', C=c)
        if model_type == 'SVM':
            if kernel is None:
                model = SVC(probability=True)
            else:
                model = SVC(kernel=kernel, gamma=gamma, C=c, probability=True)
        if model_type == 'knn':
            if kernel is None:
                model = KNeighborsClassifier(n_neighbors=neighbors)
            else:
                model = KNeighborsClassifier(kernel=kernel, gamma=gamma, C=c, probability=True)
    else:
        model = model_type

    ## Fit model with training set ##
    fitted_model = model.fit(X_train, y_train)
    
    ## Make predictions with training and test set ##
    y_pred_train, y_pred_test = fitted_model.predict(X_train), fitted_model.predict(X_test)
    if model_type != 'lin_reg':
        y_prob_train, y_prob_test = [x[1] for x in model.predict_proba(X_train)], [x[1] for x in model.predict_proba(X_test)]
        predictions_df = pd.DataFrame([[x[0] for x in y_train], [x[0] for x in y_test], list(y_pred_train), list(y_pred_test), y_prob_train, y_prob_test],
                                      index=['ytrue_train','ytrue_validate','ypred_train','ypred_validate','prob1_train','prob1_validate'])
    else:
        predictions_df = pd.DataFrame([[x[0] for x in y_train], [x[0] for x in y_test], list(y_pred_train), list(y_pred_test)],
                                      index=['ytrue_train','ytrue_validate','ypred_train','ypred_validate'])

    scores_dict = {}
    for dataset, label in zip([[y_train, y_pred_train], [y_test, y_pred_test]], ['train','validate']):
        y_true = dataset[0]
        y_pred = dataset[1]
        if model_type == 'lin_reg':
            adjusted_r2 = calculate_adjusted_R2(r2_score(y_true, y_pred), nsamples, nfeatures)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            scores_dict[f'adjusted_r2_{label}'] = [adjusted_r2]
            scores_dict[f'rmse_{label}'] = [rmse]

        elif (model_type in ['log_reg','naive_bayes','SVM','knn']) | (type(model_type) != str): ## fix this section to be able to include decision tree model object
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            scores_dict[f'precision_{label}'] = [precision]
            scores_dict[f'recall_{label}'] = [recall]
            scores_dict[f'accuracy_{label}'] = [acc]
    scores_df = pd.DataFrame(scores_dict).T

    return predictions_df, scores_df, fitted_model


# In[3]:
def run_cv(k, train, test, X, y, model_type, nfeatures, nsamples, penalty, kernel, gamma, c, neighbors):
    ## Define the model and model_name ##
    model_name = f'fold_{k + 1}'

    ## Split data into training and testing, according ot the indexes of the current fold ##
    X_train, X_test, y_train, y_test = X[train], X[test], y[train].reshape(-1, 1), y[test].reshape(-1, 1)

    ## Extract scores and fitted model using the function defined above ##
    predictions_df, scores_df, model_fit = train_validate(model_type, X_train, X_test, y_train, y_test,
                                                          nfeatures, nsamples, model_name,
                                                          penalty=penalty, kernel=kernel, gamma=gamma, c=c,
                                                          neighbors=neighbors)
    return predictions_df, scores_df, model_fit, model_name

def pipeline(model_type, kfold, X, y, penalty=None, kernel=None, gamma=None, C=[1], neighbors=5):
    tuning_predictions = {}
    tuning_scores = {}
    tuning_models = {}
    for c in C:
        ## Initialize empty containers to append results from each fold ##
        models = {} #for model objects, in case we need to save the model and access them later
        all_scores = {} #for accuracy scores
        all_predictions = {}
        nfeatures = X.shape[1]
        nsamples = X.shape[0]

        ## Loop through sklearn kfold object to access the indexes of each fold ##
        if 'Stratified' in str(type(kfold)):
            for k, (train, test) in enumerate(kfold.split(X, y)):
                predictions_df, scores_df, model_fit, model_name = run_cv(k, train, test, X, y,
                                                              model_type, nfeatures, nsamples,
                                                              penalty, kernel, gamma, c,
                                                              neighbors)

                ## Append fitted model and scores to empty containers outside the loop ##
                models[model_name] = model_fit
                all_scores[model_name] = scores_df
                all_predictions[model_name] = predictions_df
        else:
            for k, (train, test) in enumerate(kfold.split(X)):
                predictions_df, scores_df, model_fit, model_name = run_cv(k, train, test, X, y,
                                                              model_type, nfeatures, nsamples,
                                                              penalty, kernel, gamma, c,
                                                              neighbors)

                ## Append fitted model and scores to empty containers outside the loop ##
                models[model_name] = model_fit
                all_scores[model_name] = scores_df
                all_predictions[model_name] = predictions_df

        ## Create a dataframe of the accuracy scores from each fold ##
        all_scores = pd.concat(all_scores, axis=1).T.droplevel(1)

        ## Create a dataframe of the predictions from each fold ##
        all_predictions = pd.concat(all_predictions, axis=1).T.droplevel(1)

        tuning_predictions[c] = all_predictions
        tuning_scores[c] = all_scores
        tuning_models[c] = models

    tuning_predictions = pd.concat(tuning_predictions)
    tuning_predictions.index.names = ['C','fold']
    tuning_scores = pd.concat(tuning_scores)
    tuning_scores.index.names = ['C','fold']
    
    return tuning_predictions, tuning_scores, tuning_models


# In[4]:
def plot_metric(metric_df, metric, title, ax_obj, col):
    for dataset, color in zip(['train', 'validate'], ['indianred', 'dodgerblue']):
        dataset_df = metric_df.T[metric_df.T.index.str.contains(dataset)].T
        ax_obj.plot(dataset_df.values,
                     label=f'CV ({dataset})',
                     marker='o', color=color)
        ax_obj.axhline(dataset_df.mean().values,
                        label=f'CV ({dataset} avg.)',
                        linestyle='--', color=color, alpha=0.3)

    ax_obj.set_title(f'{metric}', fontsize=14)
    ax_obj.set_xticks(list(range(metric_df.shape[0])))
    ax_obj.set_xticklabels(metric_df.index.get_level_values(1), rotation=45, ha='right')
    # ax_obj.legend() if col == 0 else None


def plot_weights(models, feature_name, type_feature_names, all_feature_names, x, ax_obj, color=None, cat_feature=None):
    coeff_idx = np.where(all_feature_names == feature_name)[0]
    feature_coeffs = []
    for model_name, model_object in models.items():
        coeff_values = model_object.coef_.reshape(-1)
        feature_coeff = coeff_values[coeff_idx]
        ax_obj.scatter(x, feature_coeff,
                      edgecolor='black',facecolor='None',
                      alpha=0.5, s=100,
                      zorder=2)
        feature_coeffs.append(feature_coeff)
        
    if color == None:
        ax_obj.bar(x, np.mean(np.array(feature_coeffs)),
                  alpha=0.7, 
                  zorder=1)
    else:
        ax_obj.bar(x, np.mean(np.array(feature_coeffs)),
                  alpha=0.7, 
                  zorder=1,
                  color=color,
                  label=cat_feature)
        
    ax_obj.set_xticks(list(range(len(type_feature_names))))
    ax_obj.set_xticklabels(type_feature_names, rotation=45, ha='right')


# In[5]:


def subplot_weights(model, all_feature_names, numerical_feature_names, categorical_class_names, numeric_ratio, categorical_ratio, categorical_colors, title):
    fig, ax = plt.subplots(nrows=1, ncols=2,
                           figsize=(18,5),
                           sharey=True,
                           gridspec_kw={'width_ratios':[numeric_ratio, categorical_ratio],
                                        'wspace':0.1})

    for x, (num_feature) in enumerate(numerical_feature_names):
        plot_weights(model, num_feature, numerical_feature_names, all_feature_names, x, ax[0])

    x = 0
    legends = []
    for cat_feature, color in categorical_colors.items():
        cat_classes = [_ for _ in categorical_class_names if cat_feature in _]
        for i, (cat_class) in enumerate(cat_classes):
            if i == 0:
                plot_weights(model, cat_class, categorical_class_names, all_feature_names, x, ax[1], color=color, cat_feature=cat_feature)
            else:
                plot_weights(model, cat_class, categorical_class_names, all_feature_names, x, ax[1], color=color)

            x += 1
    ax[0].set_title('Numerical features')
    ax[0].set_ylabel('Feature weight')
    ax[1].set_title('Classes within each categorical feature (color coded by feature)')
    ax[1].legend()
    fig.suptitle(title, fontsize=25, y=1.05);


# In[6]:


def apply_feature_selection(model_type, models_to_test, X_df, y, kfold):
    nmodels = len(models_to_test.keys())
    fig, ax = plt.subplots(nrows=1, ncols=4,
                           sharey=False,
                           figsize=(12,3))
    mins = {}
    fitted_models = {}
    all_scores = {}
    for x_loc, (model_name, features_to_remove) in enumerate(models_to_test.items()):
        classes_to_remove = [c for c in X_df.columns for f in features_to_remove if f in c]
        filtered_X_df = X_df.drop(classes_to_remove, axis=1)
        X_filtered = filtered_X_df.to_numpy()
        predictions, scores, model_fit = pipeline(model_type, kfold, X_filtered, y)
        fitted_models[model_name] = model_fit
        all_scores[model_name] = scores

        for col, (col_name, col_data) in enumerate(scores.items()):
            ax[col].scatter([x_loc]*5, col_data,
                            edgecolor='black', facecolor='None',
                            zorder=2)
            ax[col].bar(x_loc, col_data.mean(),
                        alpha=0.7,
                        zorder=1,
                        label=model_name)
            ax[col].set_xticks(list(range(nmodels)))
            ax[col].set_xticklabels(models_to_test.keys(),
                                    rotation=45, ha='right')
            ax[col].set_title(col_name)

            if col not in mins.keys():
                mins[col] = []
                mins[col].append(col_data.min())
            else:
                mins[col].append(col_data.min())

    for col, col_min in mins.items():
        ymax = ax[col].get_ylim()[1]
        ax[col].set_ylim(min(col_min)*0.8, ymax)
        
    return all_scores, fitted_models


# In[7]:


def input_feature_matrixes(model_type, models_to_test, kfold):
    nmodels = len(models_to_test.keys())
    fig, ax = plt.subplots(nrows=1, ncols=4,
                           sharey=False,
                           figsize=(12,3))
    mins = {}
    fitted_models = {}
    for x, (model_name, df) in enumerate(models_to_test.items()):
        y = df['pre_score'].to_numpy()
        X = df.drop('pre_score', axis=1).to_numpy()
        predictions, scores, model_fit = pipeline(model_type, kfold, X, y)
        fitted_models[model_name] = model_fit

        for col, (col_name, col_data) in enumerate(scores.items()):
            ax[col].scatter([x]*5, col_data,
                            edgecolor='black', facecolor='None',
                            zorder=2)
            ax[col].bar(x, col_data.mean(),
                        alpha=0.7,
                        zorder=1,
                        label=model_name)
            ax[col].set_xticks(list(range(nmodels)))
            ax[col].set_xticklabels(models_to_test.keys(),
                                    rotation=45, ha='right')
            ax[col].set_title(col_name)

            if col not in mins.keys():
                mins[col] = []
                mins[col].append(col_data.min())
            else:
                mins[col].append(col_data.min())

    for col, col_min in mins.items():
        ymax = ax[col].get_ylim()[1]
        ax[col].set_ylim(min(col_min)*0.8, ymax)
        
    return scores, fitted_models


def ffs(model_type, X_df, X_pre_processed_df, y, kfold, verbose):
    included = []
    included_encoded = []
    # keep track of model and parameters
    best = {'feature': '', 'a_r2': 0, 'fitted model': None}
    while True:
        changed = False

        # list the features to be evaluated
        excluded = list(set(X_df.columns) - set(included))

        # for each remaining feature to be evaluated
        for new_column in excluded:
            cols = included + [new_column]
            classes_to_include = [c for c in X_pre_processed_df.columns for f in cols if f in c]
            X = X_pre_processed_df[classes_to_include].to_numpy()

            predictions, scores, model_fit = pipeline(model_type, kfold, X, y)
            adjusted_r2 = scores['adjusted_r2_validate'].mean()

            # if model improves
            if adjusted_r2 > best['a_r2']:
                # record new parameters
                best = {'feature': new_column, 'a_r2': adjusted_r2, 'fitted model': model_fit}
                # flag that found a better model
                changed = True

        # if found a better model after testing all remaining features
        if changed:
            # update control details
            included.append(best['feature'])
            excluded = list(set(excluded) - set(best['feature']))

            if verbose == True:
                print('Added feature %-4s with adjusted R^2 = %.3f' %
                      (best['feature'], best['a_r2']))
        else:
            # terminate if no better model
            break

    included_encoded = [c for c in X_pre_processed_df.columns for f in included if f in c]
    print('*' * 80)
    print(f'Best model has {len(included)} features:\n{included}')
    print()
    print(f'Best model has {len(included_encoded)} encoded features:\n{included_encoded}')

    return included, included_encoded
# In[ ]:




