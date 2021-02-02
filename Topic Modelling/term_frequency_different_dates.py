# -*- coding: utf-8 -*-

import pandas as pd
import sys
import numpy as np
import string
from string import digits
import re
from gensim.parsing.preprocessing import remove_stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from sklearn.decomposition import LatentDirichletAllocation
from collections import defaultdict
import operator
import time


user_nOfCooccurrence = input('Please enter number of co-occurrent word sets(recommended 2-5):\n')
user_cooccurrence_threshold = input('Please enter threshold value for co-occurrent terms to be frequent(recommended 0.07-0.12) :\n')
user_nOfTopics = input('Please enter number of topics to infer(recommended 1-3, max 4):\n')
user_nOfWords = input('Please enter number of topic words(recommended 1-3, max 5):\n')
# If user doesn't enter a number, set variables user_nOfTopics = 1, user_nOfWords = 2,  user_nOfCooccurrence = 3

if(user_nOfTopics != ''):
    try:
        user_nOfTopics = int(user_nOfTopics)
        if(user_nOfTopics > 4):
            print('Too large number set for number of topics, number of topics is set to 1')
            user_nOfTopics = 1
    except:
        print('Input is not valid, number of topics is set to 1')
        user_nOfTopics = 1
else:
    user_nOfTopics = 1

if(user_cooccurrence_threshold != ''):
    try:
        user_cooccurrence_threshold = float(user_cooccurrence_threshold)
        if(user_cooccurrence_threshold > 1):
            print('Threshold value cannot be greater than 1, it is set to 0.1 by default.')
            user_cooccurrence_threshold = 0.1
    except:
        print('Input is not valid, threshold value is set to 0.1')
        user_cooccurrence_threshold = float(0.1)
else:
    user_cooccurrence_threshold = float(0.1)

if(user_nOfWords != ''):
    try:
        user_nOfWords = int(user_nOfWords)
        if(user_nOfWords > 5):
            print('Too large number set for number of topic words, number of topic words is set to 2')
            user_nOfWords = 2
    except:
        print('Input is not valid, number of topic words is set to 2')
        user_nOfWords = int(2)
else:
    user_nOfWords = 2
    
if(user_nOfCooccurrence != ''):
    try:
        user_nOfCooccurrence = int(user_nOfCooccurrence)
    except:
        print('Input is not valid, number of co-occurrent word sets is set to 3')
        user_nOfCooccurrence = 3
else:
    user_nOfCooccurrence = 3
    
start_time = time.time()
porter = PorterStemmer()
# To be removed list that does not exist in stopwords or string.punctuations
to_be_removed = (['its', 'much','dont','th','ve','“','’','…','“','’','”','”',',','‘','—','🦠','🤴','🤔','🟩','🤣'])
strange_punctuations = (['“','’','…','“','’','”','”',',','‘','—','🦠','🤴','🤔','🟩','🤣'])
# Function to remove emojis
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
    stemWords = []
    for word in tokens:
        stemWords.append(porter.stem(word))
        stemWords.append(" ")
    return stemWords
    
# Function to remove mentions, URLs, punctuation, English stopwords and emojis from text
def preprocess(text,to_be_removed):
    t_replaced = []
    text = text.str.lower()
    tmp = ""
    for i in range(0,len(text)):
        # for debugging purposes I create original_t to compare the original text and the preprocessed 
        # text and check what is omitted at each step, I had a TypeError with some tweets, that is why
        # I created a try/except block to see in which index the error occurred, I realized that 
        # some tweets only had hashtags in them so the text part was nan in the FIFA dataset
        # I removed those rows to fix the problem
        try:
            original_t = text[i]
            tmp = re.sub('@[A-Za-z0-9]+', '',text[i])
            # Remove URLs
            tmp = re.sub(r"http\S+","",tmp)
            tmp = remove_emoji(tmp)
            # Remove triple dots
            tmp = re.sub("…","",tmp)
            # Remove punctuations and numbers, I noticed that if I use str.maketrans('', '', punctuation)
            # since some users didn't leave a whitespace after the punctuation, sometimes two words get
            # together and we miss them in our word count,         
            tmp = tmp.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
            tmp = tmp.translate(str.maketrans('','', digits))
            tmp = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', tmp) 
            tmp = remove_stopwords(tmp)
            last_check = [word for word in tmp.split() if word not in to_be_removed]
            tmp = ' '.join(last_check)
            for j in range(0,len(strange_punctuations)):
                  tmp = re.sub(strange_punctuations[j],"",tmp)
            tmp = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', tmp) 
            # Add the preprocessed final form of text to list
            t_replaced.append(tmp)
        except:
            print("Oops!", sys.exc_info()[0], "occurred.")
            print("Index ", i, " had a problem.")
    return t_replaced

def find_cooccuring(df_combined,nOfTerms):
    df_combined = df_combined.reset_index(drop=True)
    com = defaultdict(lambda : defaultdict(int))
# Forming the co-occurence matrix of single terms
    for line in range(0,len(df_combined)):
        tweet = df_combined.l_text_stemmed[line]
        date = df_combined.date[line].strftime('%d/%m/%Y')
        single_terms = tweet.split()
        # Build co-occurrence matrix
        for i in range(len(single_terms)-1):            
            for j in range(i+1, len(single_terms)):
                word1, word2 = sorted([single_terms[i], single_terms[j]])                
                if word1 != word2:
                    com[word1][word2] += 1

    com_maximum = []
    # For each term, look for the most common co-occurrent terms
    for t1 in com:
        t1_max_terms = sorted(com[t1].items(), key=operator.itemgetter(1), reverse=True)[:10]
        for t2, t2_count in t1_max_terms:
            frequency = round((t2_count/len(df_combined)),2)
            com_maximum.append(((t1, t2), t2_count, date, frequency))
    # Finding the most frequent co-occurrent terms
    max_cooccur = sorted(com_maximum, key=operator.itemgetter(1), reverse=True)
    print (max_cooccur[:nOfTerms])
    return (max_cooccur[:nOfTerms])

def find_topics(df, nOfTopics, nOfWords):
    df = df.reset_index(drop=True)
    all_topics = []
    vectorizer = CountVectorizer(max_df=0.9, min_df=25, token_pattern='\w+|\$[\d\.]+|\S+')
    text_sample = df.l_text_stemmed
    date_sample = df.date
    # apply transformation
    tf = vectorizer.fit_transform(text_sample)
            
    # tf_feature_names tells us what word each column in the matrix represents
    tf_feature_names = vectorizer.get_feature_names()
    # Define random state for reproducibility
    model = LatentDirichletAllocation(n_components=nOfTopics, random_state=0)
            
    topic_model = model.fit(tf)

    all_topics.append(display_topics(topic_model, tf_feature_names, nOfWords, date_sample))

    print(all_topics)
    return all_topics        

def display_topics(model, feature_names, no_top_words, date):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d date" % (topic_idx)] = ['{}'.format(date)
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d words" % (topic_idx)]= ['{}'.format(feature_names[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)]= ['{:.1f}'.format(topic[i])
                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)

print('Reading csv files...\n')    
df = pd.read_csv('covid19_tweets.csv')
df = df[['text','date']]   
df2 = pd.read_csv('vaccination_tweets.csv')
df3 =  pd.read_csv('FIFA.csv')
df3 = df3[:(int(len(df3)/3))]
df2 = df2[['text','date']]
df3 = df3[['Tweet','Date']]
df3.columns = ['text','date']
df2 = df2.reset_index(drop = True)
df3 = df3.reset_index(drop = True)
df2['new_index'] = range(len(df), len(df)+len(df2), 1)

df2 = df2.set_index('new_index')

df3['new_index'] = range(len(df)+len(df2), len(df)+len(df2)+len(df3), 1)

df3 = df3.set_index('new_index')
df_combined = pd.concat([df,df2,df3], axis = 0, ignore_index = True)

# Remove rows with nan values in the text column
df_combined = df_combined.dropna(subset=['text'])

df_combined = df_combined.reset_index(drop = True)

print('Starting preprocessing...\n')
preprocess_start_time = time.time()
df_combined.text = preprocess(df_combined.text, to_be_removed) 
preprocess_end_time = round((time.time() - preprocess_start_time),2)
print('Preprocessing complete...\n')
print('Total execution time for preprocessing: ',preprocess_end_time,' seconds.\n')
print('Starting tokenization and stemming...\n')
stemming_tokenization_start_time = time.time()
text_tokenized_stemmed = df_combined.text.apply(tokenizer)
stemming_tokenization_end_time = round((time.time() - stemming_tokenization_start_time),2)
print('Stemming complete...\n')
print('Total execution time for tokenization and stemming: ',stemming_tokenization_end_time,' seconds.\n')


# Create a big list of words
long_text_stemmed = [' '.join(row) for row in text_tokenized_stemmed]
df_combined['l_text_stemmed'] = long_text_stemmed

df_combined['date'] = pd.to_datetime(df_combined['date'])
df_combined['date'] = df_combined['date'].dt.date

# Check for unique dates
tweet_dates =  np.unique(df_combined.date)

# Create a dataframe for results
result_df = pd.DataFrame()
all_topics = pd.DataFrame()
cooccurrence_end_time = 0
topic_model_end_time = 0
for i in range(0,len(tweet_dates)): 
    try:
        cooccurrence_start_time = time.time()
        vectorizer = CountVectorizer(max_df=0.9, min_df=25, token_pattern='\w+|\$[\d\.]+|\S+')       
        # Get the tweets from different dates    
        temp_df = df_combined[df_combined.date.apply(lambda x: x.strftime('%d/%m/%Y'))==tweet_dates[i].strftime("%d/%m/%Y")]
        # Find co-occurring terms on the given date
        print('***** Co-occurring terms on ',tweet_dates[i].strftime('%d/%m/%Y'),' *****\n')
        result_df = result_df.append(find_cooccuring(temp_df,user_nOfCooccurrence))
        cooccurrence_end_time += round((time.time() - cooccurrence_start_time),2)
        print('\n')
        # Find the prevalent topics on the given date
        topic_model_start_time = time.time()
        print('***** Topics inferred on ',tweet_dates[i].strftime('%d/%m/%Y'),'*****\n')
        all_topics = all_topics.append(find_topics(temp_df, user_nOfTopics, user_nOfWords))
        topic_model_end_time += round((time.time() - topic_model_start_time),2)
        print('\n')
    except:
        print('No topics found for ', tweet_dates[i],'\n')
        continue
    
# Rename columns of the result dataframe    
result_df.columns = ['Co-occurring_terms','Count','Date','Frequency']

# Check the number of unique co-occurring terms
len(np.unique(result_df['Co-occurring_terms']))

# Group by co-occurring terms to see on which dates they were popular together
result_df2 = result_df.groupby(['Co-occurring_terms'])['Date'].apply(lambda x: ' '.join(x)).reset_index()

# Find the co-occurring terms that co-occurred more than a single day
freq_terms = []
for i in range(0,len(result_df2)):
    if(len(result_df2['Date'][i])>10):
        freq_terms.append(result_df2['Co-occurring_terms'][i])
        
#threshold = 0.1
result_high_freq = result_df[result_df['Frequency']>user_cooccurrence_threshold]
result_high_freq = result_high_freq.reset_index(drop=True)
unique_high_freq_terms = np.unique(result_high_freq['Co-occurring_terms'])
print('***** Co-occurrent Terms with Frequency above Threshold *****\n\n',result_high_freq,'\n')
total_end_time = round((time.time() - start_time),2)


print('Total execution time for co-occurrent terms: ',round(cooccurrence_end_time,2),' seconds.\n')
print('Total execution time for topic model: ',round(topic_model_end_time,2),' seconds.\n')
print('Total execution time: ',total_end_time,' seconds.\n')

k=input("press a key to exit") 
print('Exiting...')






