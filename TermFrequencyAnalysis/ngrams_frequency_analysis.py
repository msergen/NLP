import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import string
from string import digits, punctuation
import re
import gensim
from gensim.parsing.preprocessing import remove_stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer

to_be_removed = (['its', 'much','dont','th','ve'])
strange_punctuations = (['“','’','…','“','’','”','”',',','‘','—','🦠','🤴','🤔','🟩','🤣'])

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)
	
def tokenizer(text):
    tokens = nltk.word_tokenize(text)
    return tokens

def preprocess(text,to_be_removed):
    t_replaced = []
    text = text.str.lower()
    tmp = ""
    for i in range(0,len(text)):
        # for debugging purposes I create original_t to compare the original text and the preprocessed 
        # text and check what is omitted at each step
        original_t = text[i]
        # Remove URLs
        tmp = re.sub(r"http\S+","",tmp)
        tmp = remove_emoji(tmp)
        # Remove punctuations and numbers, I noticed that if I use str.maketrans('', '', punctuation)
        # since some users didn't leave a whitespace after the punctuation, sometimes two words get
        # together and we miss them in our word count,         
        tmp = tmp.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
#       tmp = tmp.translate(str.maketrans('', '', punctuation))
        tmp = tmp.translate(str.maketrans('','', digits))
        tmp = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', tmp) 
        tmp = remove_stopwords(tmp)
        last_check = [word for word in tmp.split() if word not in to_be_removed]
        tmp = ' '.join(last_check)
        for j in range(0,len(strange_punctuations)):
              tmp = re.sub(strange_punctuations[j],"",tmp)
        # Add the preprocessed final form of text to list
        t_replaced.append(tmp)
    return t_replaced

def top_n_bitri_grams(corpus, n=None):
    v = CountVectorizer(ngram_range=(2,3))
    X = v.fit_transform(corpus)
    bi_tri_df = pd.DataFrame(X.toarray(), columns = v.get_feature_names())
    bi_tri_summed = bi_tri_df.sum()
    top_n = pd.DataFrame({'ngram': bi_tri_summed.index, 'count':  bi_tri_summed.values})
    top_n = top_n.sort_values(by='count', ascending=False).head(n)
    return top_n	
	
df = pd.read_csv('~/covid19_tweets.csv')

df2 = df[['user_location', 'text', 'hashtags']]

df2.text = preprocess(df2.text, to_be_removed)

# Tokenize each row
text_tokenized = df2.text.apply(tokenizer)

# Create a big list of words
long_text = [' '.join(row) for row in text_tokenized]

df2.text = long_text
long_string = ' '.join(long_text)

# Get the frequency distribution
fd = nltk.FreqDist([word for word in long_string.split()])

# Converting FreqDist dictionary to Pandas dataframe
df_fdist = pd.DataFrame.from_dict(fd, orient='index')
df_fdist.columns = ['Count']
df_fdist.index.name = 'Unigrams'

# Sorting the data frame in descending order
df_fdist = df_fdist.sort_values(['Count'], ascending = False)

# Visualize the bar plot
df_fdist['Count'].head(20).plot(kind='bar')

# Plot without the word 'covid'
df_fdist['Count'][1:].head(20).plot(kind='bar')

top_n = pd.DataFrame(data = None)
# Divide in smaller dataframes to save memory
for i in range(0,round(len(df2.text)/10000)):
    if(i!=round(len(df2['text'])/10000)-2):
        top_10_bi_tri = top_n_bitri_grams(df2['text'][i*10000:((i+1)*10000)+1],20)
    else:
        top_10_bi_tri = top_n_bitri_grams(df2['text'][i*10000:len(df2['text'])+1],20)
    # Gather all top 20 bigrams and trigrams 
    top_n = top_n.append(top_10_bi_tri)

# create a copy of top_n to avoid running the code again if any problems are encountered    
copy_counts = top_n

# Group by ngram, sum their counts and sort by counts in descending order, show top 20
group_by_ngram = copy_counts.groupby('ngram')['count'].sum()
group_by_ngram = group_by_ngram.sort_values(ascending = False)
group_by_ngram = group_by_ngram[:20]

# Plot most frequent bigrams(since there was no trigram in the top 20 list)
plt.figure(figsize=(20,8))
fig = group_by_ngram.plot(kind='bar')
fig.set_title('Most Frequent Bigrams')
fig.set_xlabel('bigrams')
fig.set_ylabel('count')
