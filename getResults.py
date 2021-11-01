import re
import os
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
import string

from data_preprocessing import y_train, y_test, train, test
from data_preprocessing import stops, X_train, X_test

import config


def get_nonexistant_path(filename):  # https://stackoverflow.com/a/43167607/14598178
    """
    Get the path to a filename which does not exist by incrementing path.
    """

    if not os.path.exists(filename):
        return filename
    filename, file_extension = os.path.splitext(filename)
    i = 1
    new_filename = "{}-{}{}".format(filename, i, file_extension)
    while os.path.exists(new_filename):
        i += 1
        new_filename = "{}-{}{}".format(filename, i, file_extension)
    return new_filename


def getResults(algoList, verbose=True, save=False):
    """
    :param algoList: List of gridSearch-class objects you want to get the results for
    :param verbose: If true prints the final Pandas dataframe
    :param save: If true saves the file as a .csv file in '/results/{dataset name}_results.csv'
    :return:
    """
    beforeParenthesis = re.compile("(.*?)\s*\(")

    dfResults = pd.DataFrame(columns=['Classifier', 'Parameters', 'Train Accuracy', 'Test Accuracy',
                                      'Precision', 'Recall', 'F1-Score', 'CV Splits', 'Training time',
                                      'ngram_range', 'min_df', 'stemming', 'lemmatization', 'lowercase',
                                      'remove_stops'])
    for algo in algoList:
        tempList = []

        # model things
        classifier = beforeParenthesis.match(str(algo.best_estimator_)).group(1)
        tempList.append(classifier)

        parameters = list(algo.best_params_.items())
        tempList.append(parameters)

        train_accuracy = algo.score(train, y_train)
        tempList.append(train_accuracy)

        test_accuracy = algo.best_score_
        tempList.append(test_accuracy)

        y_true, y_pred = y_test, algo.predict(test)
        precision, recall, fScore, support = precision_recall_fscore_support(y_true, y_pred, average='macro')
        tempList.append(precision)
        tempList.append(recall)
        tempList.append(fScore)

        splits = algo.n_splits_
        tempList.append(splits)

        trainTime = algo.refit_time_
        tempList.append(trainTime)

        # config things
        tempList.append(config.ngram_range)
        tempList.append(config.min_df)
        tempList.append(config.stemming)
        tempList.append(config.lemmatization)
        tempList.append(config.lowercase)
        tempList.append(config.remove_stops)

        dfResults.loc[len(dfResults)] = tempList

    if save:
        os.chdir(r'D:/PycharmProjects/INFOMDM-2021-P2/results')
        dfResults.to_csv(get_nonexistant_path('results.csv'))
        os.chdir(r'D:/PycharmProjects/INFOMDM-2021-P2')

    if verbose:
        print(dfResults)


# Number of words in the text
X_train_num_words = X_train.apply(lambda x: len(str(x).split()))
X_test_num_words = X_test.apply(lambda x: len(str(x).split()))

# Number of unique words in the text
X_train_num_unique_words = X_train.apply(lambda x: len(set(str(x).split())))
X_test_num_unique_words = X_test.apply(lambda x: len(set(str(x).split())))

# Number of characters in the text
X_train_num_chars = X_train.apply(lambda x: len(str(x)))
X_test_num_chars = X_test.apply(lambda x: len(str(x)))

# Number of stopwords in the text
X_train_num_stopwords = X_train.apply(lambda x: len([w for w in str(x).lower().split() if w in stops]))
X_test_num_stopwords = X_test.apply(lambda x: len([w for w in str(x).lower().split() if w in stops]))

# Number of punctuations in the text
X_train_num_punctuation = X_train.apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
X_test_num_punctuation = X_test.apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

# Number of title case words in the text
X_train_num_words_upper = X_train.apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
X_test_num_words_upper = X_test.apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

# Number of title case words in the text
X_train_num_words_title = X_train.apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
X_test_num_words_title = X_test.apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

# Average length of the words in the text
X_train_mean_word_len = X_train.apply(lambda x: np.mean([len(w) for w in str(x).split()]))
X_test_mean_word_len = X_test.apply(lambda x: np.mean([len(w) for w in str(x).split()]))
