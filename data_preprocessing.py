from sklearn.feature_extraction.text import CountVectorizer
from data_in import df_training, df_testing

# get train and test X/y splits
X_train = df_training['observation']
y_train = df_training['label'].astype('int')
X_test = df_testing['observation']
y_test = df_testing['label'].astype('int')

# Create a Counter of tokens
cv = CountVectorizer(decode_error='strict', lowercase=True, min_df=1)  # we could change min_df and see differences
train = cv.fit_transform(X_train)
test = cv.transform(X_test)

# print('Train size: ', train.shape)
# print('Test size: ', test.shape)

# vocab = list(cv.vocabulary_.items())
# print(vocab[:10])
