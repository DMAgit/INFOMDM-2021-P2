import re
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd


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

        y_true, y_pred = y_test, algo.predict(X_test)
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
