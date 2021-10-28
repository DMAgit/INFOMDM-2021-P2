# Config for the data preprocessing

# CountVectorizer
ngram_range = (1, 1)  # The lower and upper boundary of the range of n-values for different word n-grams or char n-grams to be extracted.
min_df = 1

# Word normalization
stemming = False
lemmatization = False
lowercase = True
remove_stops = False
