from mlxtend.evaluate import paired_ttest_5x2cv

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

import data_preprocessing


# statistical test to
# check if difference between algorithms is significant
# adapted from http://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_5x2cv/
def t_test(estimator1, estimator2, X, y):
    t, p = paired_ttest_5x2cv(estimator1, estimator2, X, y, scoring='accuracy', random_seed=42)
    # summarize
    print(f'The P-value is = {p:.4f}')
    print(f'The t-statistics is = {t:.4f}')
    # interpret the result
    if p <= 0.05:
        print(
            'Since p<0.05, We can reject the null-hypothesis that both models perform equally well on this dataset. We '
            'may conclude that the two algorithms are significantly different.')
    else:
        print('Since p>0.05, we cannot reject the null hypothesis and may conclude that the performance of the two '
              'algorithms is not significantly different.')


# implement the models with the best hyperparameters

# Naive Bayes
# set the configuration to the one which gave the best performance for each classifier
ngram_range = (1, 1)
stemming = True
lemmatization = False
lowercase = True
remove_stops = True

# pre-process the data
# clean text
X_train = data_preprocessing.X_train.map(lambda x: data_preprocessing.cleanData(
    x, lowercase, remove_stops, stemming, lemmatization))

# Create a Counter of tokens
cv = CountVectorizer(ngram_range=ngram_range, min_df=1)  # we could change min_df and see differences
train = cv.fit_transform(X_train)

nb = MultinomialNB(alpha=2.5)
nb.fit(train, data_preprocessing.y_train)

# Logistic Regression
# set the configuration to the one which gave the best performance for each classifier
ngram_range = (1, 2)
stemming = True
lemmatization = False
lowercase = True
remove_stops = True

# pre-process the data
# clean text
X_train = data_preprocessing.X_train.map(lambda x: data_preprocessing.cleanData(
    x, lowercase, remove_stops, stemming, lemmatization))

# Create a Counter of tokens
cv = CountVectorizer(ngram_range=ngram_range, min_df=1)  # we could change min_df and see differences
train = cv.fit_transform(X_train)

lr = LogisticRegression(C=0.1, penalty='l2')
lr.fit(train, data_preprocessing.y_train)

# Decision Tree
# set the configuration to the one which gave the best performance for each classifier
ngram_range = (1, 1)
stemming = False
lemmatization = False
lowercase = True
remove_stops = True

# pre-process the data
# clean text
X_train = data_preprocessing.X_train.map(lambda x: data_preprocessing.cleanData(
    x, lowercase, remove_stops, stemming, lemmatization))

# Create a Counter of tokens
cv = CountVectorizer(ngram_range=ngram_range, min_df=1)  # we could change min_df and see differences
train = cv.fit_transform(X_train)

dt = DecisionTreeClassifier(max_depth=2, min_samples_split=4)
dt.fit(train, data_preprocessing.y_train)

# Random Forest
# set the configuration to the one which gave the best performance for each classifier
ngram_range = (1, 2)
stemming = True
lemmatization = False
lowercase = True
remove_stops = True

# pre-process the data
# clean text
X_train = data_preprocessing.X_train.map(lambda x: data_preprocessing.cleanData(
    x, lowercase, remove_stops, stemming, lemmatization))

# Create a Counter of tokens
cv = CountVectorizer(ngram_range=ngram_range, min_df=1)  # we could change min_df and see differences
train = cv.fit_transform(X_train)

rf = RandomForestClassifier(max_depth=4, min_samples_split=2, n_estimators=1000)
rf.fit(train, data_preprocessing.y_train)


# and finally:
# Check NB and LR
print("Naive Bayes vs Logistic Regression")
t_test(nb, lr, train, data_preprocessing.y_train)

print("\n\n")

# Check DT and RF
print("Decision Tree vs RF")
t_test(dt, rf, train, data_preprocessing.y_train)
