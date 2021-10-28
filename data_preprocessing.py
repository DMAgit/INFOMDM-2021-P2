from sklearn.feature_extraction.text import CountVectorizer
from data_in import df_training, df_testing
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import re

from config import ngram_range, min_df
from config import lowercase, remove_stops, stemming, lemmatization

# function to clean data
stops = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
         "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
         "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
         "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
         "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
         "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
         "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
         "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
         "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
         "should", "now"]  # stop word list taken from NLTK (https://gist.github.com/sebleier/554280)


def cleanData(text, lowercase=True, remove_stops=False, stemming=False, lemmatization=False):
    txt = str(text)

    txt = re.sub(r'[^A-Za-z\s]', r' ', txt)

    if lowercase:
        txt = " ".join([w.lower() for w in txt.split()])

    if remove_stops:
        txt = " ".join([w for w in txt.split() if w not in stops])

    if stemming:
        st = SnowballStemmer(language='english')
        txt = " ".join([st.stem(w) for w in txt.split()])

    if lemmatization:
        wordnet_lemmatizer = WordNetLemmatizer()
        txt = " ".join([wordnet_lemmatizer.lemmatize(w, pos='v') for w in txt.split()])

    return txt


# get train and test X/y splits
X_train = df_training['text']
y_train = df_training['label'].astype('int')
X_test = df_testing['text']
y_test = df_testing['label'].astype('int')

# clean text
X_train['text'] = X_train['text'].map(lambda x: cleanData(x, lowercase=lowercase, remove_stops=remove_stops, stemming=stemming, lemmatization=lemmatization))
X_test['text'] = X_test['text'].map(lambda x: cleanData(x, lowercase=lowercase, remove_stops=remove_stops, stemming=stemming, lemmatization=lemmatization))

# Create a Counter of tokens
cv = CountVectorizer(ngram_range=ngram_range, min_df=min_df)  # we could change min_df and see differences
train = cv.fit_transform(X_train)
test = cv.transform(X_test)

# print('Train size: ', train.shape)
# print('Test size: ', test.shape)

# vocab = list(cv.vocabulary_.items())
# print(vocab[:10])
