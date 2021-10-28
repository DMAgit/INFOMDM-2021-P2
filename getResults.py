import re
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
import string

from data_preprocessing import test, y_test, X_train, X_test
from data_preprocessing import stops


def getResults(algoList, verbose=True, save=False):
    """
    :param algoList: List of gridSearch-class objects you want to get the results for
    :param verbose: If true prints the final Pandas dataframe
    :param save: If true saves the file as a .csv file in '/results/{dataset name}_results.csv'
    :return:
    """
    beforeParenthesis = re.compile("(.*?)\s*\(")

    dfResults = pd.DataFrame(columns=['Classifier', 'Parameters', 'Accuracy', 'Precision', 'Recall',
                                      'F1-Score', 'CV Splits', 'Training time'])
    for algo in algoList:
        tempList = []

        classifier = beforeParenthesis.match(str(algo.best_estimator_)).group(1)
        tempList.append(classifier)

        parameters = list(algo.best_params_.items())
        tempList.append(parameters)

        accuracy = algo.best_score_
        tempList.append(accuracy)

        y_true, y_pred = y_test, algo.predict(test)
        precision, recall, fScore, support = precision_recall_fscore_support(y_true, y_pred, average='macro')
        tempList.append(precision)
        tempList.append(recall)
        tempList.append(fScore)

        splits = algo.n_splits_
        tempList.append(splits)

        trainTime = algo.refit_time_
        tempList.append(trainTime)

        dfResults.loc[len(dfResults)] = tempList

    if save:
        dfResults.to_csv(r'results/results.csv')

    if verbose:
        print(dfResults)


# Number of words in the text
X_train["num_words"] = X_train["text"].apply(lambda x: len(str(x).split()))
X_test["num_words"] = X_test["text"].apply(lambda x: len(str(x).split()))

# Number of unique words in the text
X_train["num_unique_words"] = X_train["text"].apply(lambda x: len(set(str(x).split())))
X_test["num_unique_words"] = X_test["text"].apply(lambda x: len(set(str(x).split())))

# Number of characters in the text
X_train["num_chars"] = X_train["text"].apply(lambda x: len(str(x)))
X_test["num_chars"] = X_test["text"].apply(lambda x: len(str(x)))

# Number of stopwords in the text
X_train["num_stopwords"] = X_train["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in stops]))
X_test["num_stopwords"] = X_test["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in stops]))

# Number of punctuations in the text
X_train["num_punctuations"] = X_train['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
X_test["num_punctuations"] = X_test['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

# Number of title case words in the text
X_train["num_words_upper"] = X_train["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
X_test["num_words_upper"] = X_test["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

# Number of title case words in the text
X_train["num_words_title"] = X_train["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
X_test["num_words_title"] = X_test["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

# Average length of the words in the text
X_train["mean_word_len"] = X_train["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
X_test["mean_word_len"] = X_test["text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
