import data_preprocessing
from sklearn.linear_model import LogisticRegression


"""
Use this config:
    # Config for the data preprocessing
    
    # CountVectorizer
    ngram_range = (1, 2)  # The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted.
    min_df = 1
    
    # Word normalization
    stemming = True
    lemmatization = False
    lowercase = True
    remove_stops = True
"""


lr = LogisticRegression(C=0.1, penalty='l2')
lr.fit(data_preprocessing.train, data_preprocessing.y_train)


# adapted from https://stackoverflow.com/a/11140887/14598178
def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names_out()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    print("Decepting      |      Truthful")
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print(f"{coef_1:0.4f} - {fn_1} | {coef_2:0.4f} - {fn_2}")


show_most_informative_features(data_preprocessing.cv, lr, 5)
