#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import sqlite3
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from wordcloud import WordCloud
import random
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy import displacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn import metrics


# In[3]:
def structure_data(data):
    structured_df = []
    for u in data.UserID.unique():
        u_df = data[data.UserID == u]
        u_df['AnswerText'] = u_df['AnswerText'].replace({'-1': np.nan})
        u_df = u_df.dropna()
        survey = u_df.SurveyID.unique()[0]
        unique_questions = pd.DataFrame(u_df.groupby('QuestionID')['AnswerText'].apply(list)).T.reset_index().iloc[:,
                           1:]
        unique_questions['UserID'] = u
        unique_questions['SurveyID'] = survey
        unique_questions = unique_questions.set_index(['UserID', 'SurveyID'])
        structured_df.append(unique_questions)
    structured_df = pd.concat(structured_df).T.sort_index().T

    return structured_df

def create_word_cloud(questions):
    questions_text = ''
    for idx, q in questions.iterrows():
        questions_text = questions_text+' '+q.questiontext
    questions_text = questions_text.lower() #always make sure to lower case all words since not all packages would do this in the back end
    
    # Tokenize the text
    tokens = word_tokenize(questions_text)

    # Remove punctuation and convert to lowercase
    tokens = [word for word in tokens if word.isalpha()]

    # Remove stop words
    stop_words = set(list(set(stopwords.words('english'))) + ['mental','health','issue'])
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming (using the Porter Stemmer)
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    # Lemmatization (using the WordNet Lemmatizer)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Create Word Clouds for the original and filtered text
    original = WordCloud(width=800, height=400, background_color="white").generate(" ".join(tokens))
    filtered = WordCloud(width=800, height=400, background_color="white").generate(" ".join(filtered_tokens))
    stemmed = WordCloud(width=800, height=400, background_color="white").generate(" ".join(stemmed_tokens))
    lemmatized = WordCloud(width=800, height=400, background_color="white").generate(" ".join(lemmatized_tokens))

    # Plot the Word Clouds
    fig, ax = plt.subplots(nrows=1, ncols=4,
                           figsize=(12*2, 3*2))
    for col, (cloud, title) in enumerate(zip([original, filtered, stemmed, lemmatized],
                                             ['original', 'filtered', 'stemmed', 'lemmatized']
                                            )
                                        ):
        ax[col].imshow(cloud, interpolation = 'bilinear')
        ax[col].axis('off')
        ax[col].set_title(title)
    plt.show()
    fig.savefig('word_cloud.png')

# In[4]:


def extract_CV_search_scores(CV_search_obj, n_it):
    cv_results = CV_search_obj.cv_results_

    all_train_scores = []
    all_validate_scores = []
    for i in range(n_it):
        ## Get parameters used ##
        params = cv_results['params'][i]

        train_scores = []
        validate_scores = []
        for k in range(5):
            ## Get train and validation scores ##
            train = cv_results[f'split{k}_train_score'][i]
            test = cv_results[f'split{k}_test_score'][i]

            ## Convert into dataframes ##
            train_score_k = pd.DataFrame([train], index=[f'fold{k+1}'])
            validate_score_k = pd.DataFrame([test], index=[f'fold{k+1}']) 

            ## Append dataframes to empty containers outside of the loop ##
            train_scores.append(train_score_k)
            validate_scores.append(validate_score_k)

        ## Concatenate and re-index all of the dataframes from the inner loop ##
        train_df = pd.concat(train_scores).T
        validate_df = pd.concat(validate_scores).T

        train_df.index = pd.MultiIndex.from_tuples([tuple(list(params.values())+[i])], names=list(params.keys())+['iteration'])
        validate_df.index = pd.MultiIndex.from_tuples([tuple(list(params.values())+[i])], names=list(params.keys())+['iteration'])

        ## Append dataframes to empty containers outside of the loop ##
        all_train_scores.append(train_df)
        all_validate_scores.append(validate_df)

    train_scores = pd.concat(all_train_scores)
    validation_scores = pd.concat(all_validate_scores)
    
    return train_scores, validation_scores


# In[5]:


def plot_CV_search(train_scores, validation_scores, categorical_hyperparameters, numerical_hyperparameter, numerical_scale, show_top, title):
    colormap = plt.get_cmap('plasma')
    min_reg = pd.concat([train_scores, validation_scores]).index.get_level_values(numerical_hyperparameter).min()
    max_reg = pd.concat([train_scores, validation_scores]).index.get_level_values(numerical_hyperparameter).max()
    
    for cat_key, cat_values in categorical_hyperparameters.items():
        fig, ax = plt.subplots(nrows=2, ncols=len(cat_values),
                               figsize=(5*len(cat_values),9),
                               sharey='row', sharex=False,
                               gridspec_kw={'hspace':0.5})

        for col, (cat_value) in enumerate(cat_values):
            train = train_scores[train_scores.index.get_level_values(cat_key) == cat_value]
            validate = validation_scores[validation_scores.index.get_level_values(cat_key) == cat_value]

            ## Plot train ##
            train_reg = train.index.get_level_values(numerical_hyperparameter)
            ax[0,col].scatter(train_reg, train.mean(axis=1), label='train (avg.)', c='grey')

            ## Plot validate
            val_reg = validate.index.get_level_values(numerical_hyperparameter)
            if numerical_scale == 'linear':
                ax[0,col].scatter(val_reg, validate.mean(axis=1), label='validate (avg.)', c=(val_reg-min_reg)/(max_reg-min_reg), cmap=colormap)
            else:
                ax[0,col].scatter(val_reg, validate.mean(axis=1), label='validate (avg.)', c=(val_reg-min_reg)/(max_reg-min_reg), cmap=colormap)

            ax[0,col].legend() if col == 1 else None
            ax[0,col].set_ylabel('f1') if col == 0 else None
            ax[0,col].set_xlabel(f'numeric hyperparameter: {numerical_hyperparameter}')
            ax[0,col].set_title(f'categorical hyperparameter value:{cat_value}')
            ax[0,col].set_xscale(numerical_scale)

            weighted_val = validate * (validate / train)
            sorted_val_mean = pd.DataFrame(weighted_val.mean(axis=1)).sort_values(0, ascending=False).iloc[:show_top]
            sorted_val_std = weighted_val.loc[sorted_val_mean.index].std(axis=1)
            reg = sorted_val_mean.index.get_level_values(numerical_hyperparameter)

            ax[1,col].errorbar(np.array(list(range(sorted_val_mean.shape[0]))), sorted_val_mean.values.reshape(-1), sorted_val_std.values.reshape(-1), c='black', alpha=0.5)
            ax[1,col].scatter(list(range(sorted_val_mean.shape[0])), sorted_val_mean, c=reg, cmap=colormap)

            ## Set title and axis labels ##
            ax[1,col].legend(['mean +/- std'])
            ax[1,col].set_ylabel('f1_weighted') if col == 0 else None
            ax[1,col].set_xlabel('Iteration number (ordered by avg. f1_weighted)')
            ax[1,col].set_title(f'categorical hyperparameter value:{cat_value}\n(Top {show_top} avg. f1_weighted)')
            ax[1,col].set_xticks(list(range(show_top)))
            ax[1,col].set_xticklabels([str(x) for x in sorted_val_mean.index.get_level_values('iteration')], rotation=45, ha='right')
        plt.show()
        fig.savefig(title)

def extract_metrics_from_classification_report(report_str, metric='precision'):
    class_metrics = {}

    # Split the report string into lines
    lines = report_str.split('\n')

    # Get the metric index based on the header line
    header_line = lines[0]
    header_parts = header_line.split()
    metric_index = None

    for i, header_part in enumerate(header_parts):
        if header_part.lower() == metric.lower():
            metric_index = i
            break

    if metric_index is None:
        raise ValueError(f"Metric '{metric}' not found in the classification report.")

    # Iterate through the lines starting from the 2nd line (skipping the header)
    for line in lines[2:-1]:
        parts = line.split()
        class_label = int(parts[0])  # Extract the class label as an integer
        metric_value = float(parts[metric_index])
        class_metrics[class_label] = metric_value

    return class_metrics