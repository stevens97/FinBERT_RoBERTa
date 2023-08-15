'''
# ---------------------------------------------------------
# Import Libraries
# ---------------------------------------------------------
'''
import os
import sys
import urllib
import numpy as np
import pandas as pd
import tweepy.errors
from transformers import AutoModelForSequenceClassification
from finbert.finbert import *
from utils import *
from scipy.special import softmax
import logging
logging.disable(logging.INFO)
from newsAPI import get_News
from twitterV2API import get_Tweets
import click
from tqdm import tqdm


def FinBERT():
    '''

    Create a pre-trained FinBERT (Financial Sentiment Analysis with BERT) model
    for sentiment analysis. Original documentation:

    Documentation:
    Publication: https://arxiv.org/abs/1908.10063
    Model documentation: https://huggingface.co/ProsusAI/finbert
    Source code: https://github.com/ProsusAI/finBERT

    :return:
    model: [Transformers Model]; The FinBERT model.
    '''

    finbert = "ProsusAI/finbert"
    model = AutoModelForSequenceClassification.from_pretrained(finbert)

    return model


def FinBERT_Sentiment(model, text):
    '''

    :param model: [Transformers Model]; The FinBERT model.
    :param text: [String]; The text to analyse.
    :return:
    sentiment_score: [Float]; The sentiment score of the text.
    sentiment_label: [String]; The sentiment label of the text. E.g., Negative / Neutral / Positive
    '''

    input = preprocess_text(text)
    df_sentiment = pd.DataFrame(predict(input, model))
    sentiment_score = df_sentiment['sentiment_score'].mean()
    val, count = np.unique(df_sentiment['prediction'], return_counts=True)
    idx = np.argmax(count)
    sentiment_label = str(val[idx]).title()

    return sentiment_score, sentiment_label


def RoBERTa():
    '''

    Create a pre-trained RoBERTa (A Robustly Optimized BERT Pre-training Approach) model
    for sentiment analysis.

    Documentation:
    Publication: https://arxiv.org/abs/1907.11692
    Model documentation: https://huggingface.co/docs/transformers/model_doc/roberta
    Source code: https://github.com/facebookresearch/fairseq/tree/main/examples/roberta

    :return:
    model, [Transformers Model]; The RoBERTa model.
    tokenizer, [Transformers Tokenizer]; The RoBERTa tokenizer.
    labels, [String Array]; The possible sentiment labels of the text. E.g., Negative / Neutral / Positive
    '''

    roberta = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)
    labels = ['Negative', 'Neutral', 'Positive']

    return model, tokenizer, labels


def RoBERTa_Sentiment(model, tokenizer, labels, text):
    '''

    :param model: [Transformers Model]; The RoBERTa model.
    :param tokenizer: [Transformers Tokenizer]; The RoBERTa tokenizer.
    :param labels: [String Array]; The possible sentiment labels of the text. E.g., Negative / Neutral / Positive
    :param text: [String]; The text to analyse.
    :return:
    sentiment_score: [Float]; The sentiment score of the text.
    sentiment_label: [String]; The sentiment label of the text. E.g., Negative / Neutral / Positive
    '''

    encoded_input = preprocess_text(text)
    encoded_input = tokenizer(encoded_input, return_tensors='pt')
    output = model(**encoded_input)
    scores = np.array(output[0][0].detach(), dtype=float)
    scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    for i in range(scores.shape[0]):
        l = labels[ranking[i]]
        s = scores[ranking[i]]

    negative_score = scores[labels.index('Negative')]
    neutral_score = scores[labels.index('Neutral')]
    positive_score = scores[labels.index('Positive')]

    negativity = negative_score / (negative_score + neutral_score + positive_score)
    neutrality = neutral_score / (negative_score + neutral_score + positive_score)
    positivity = positive_score / (negative_score + neutral_score + positive_score)

    sentiment_score = float((negativity * -1) + (neutrality * 0) + (positivity * 1))
    sentiment_label = labels[np.argmax(scores)].title()

    return sentiment_score, sentiment_label


def init_Models(setting):
    '''

    Initialise and returns the specified NLP models.

    :param setting: [String]; Models to initialise. Options are: 'FinBERT', 'RoBERTA'.
    :return: The necessary nlp models.
    '''

    if setting == 'FinBERT':
        FinBERT_model = FinBERT()
        return FinBERT_model
    if setting == 'RoBERTa':
        RoBERTa_model, RoBERTa_tokenizer, RoBERTa_labels = RoBERTa()
        return RoBERTa_model, RoBERTa_tokenizer, RoBERTa_labels


def sentiment(model, text):
    '''

    Use the FinBERT or RoBERTa models to evaluate the sentiment score and label of a given text.

    :param model: [String]; Specify which model you want to use by using either "FinBERT" or "RoBERTa".
    :param text: [String]; The text to analyse.
    :return:
    sentiment_score: [Float]; The sentiment score of the text.
    sentiment_label: [String]; The sentiment label of the text. E.g., Negative / Neutral / Positive
    '''

    global FinBERT_model, RoBERTa_model, RoBERTa_tokenizer, RoBERTa_labels

    if model == "FinBERT":
        score, label = FinBERT_Sentiment(FinBERT_model, text)
    if model == "RoBERTa":
        score, label = RoBERTa_Sentiment(RoBERTa_model, RoBERTa_tokenizer, RoBERTa_labels, text)

    return score, label


def news_Sentiment(df_news):
    '''

    Calculates the sentiment scores and labels for a dataframe of articles obtained with the NewsAPI.

    :param df_news [pandas DataFrame]: Dataframe of articles obtained with the NewsAPI (as returned by get_News()).
    :return df_sentiment [pandas DataFrame]: Dataframe of articles with their sentiment scores and labels.
    '''

    df_sentiment = df_news.copy()

    df_sentiment['Sentiment'] = [sentiment("FinBERT", str(x)) for x in tqdm(df_sentiment['Content'])]
    df_sentiment['Score'] = [float(x[0]) for x in df_sentiment['Sentiment']]
    df_sentiment['Label'] = [str(x[1]) for x in df_sentiment['Sentiment']]

    print('Finding top words...')
    df_sentiment['Top Words'] = [top_Words(x, 5) for x in tqdm(df_sentiment['Content'])]

    print('Finding standout articles...')
    sigma = np.abs(np.std(df_sentiment['Score']))
    avg = np.abs(np.mean(df_sentiment['Score']))
    k = [False] * len(df_sentiment)
    df_sentiment['Standout'] = k
    df_sentiment.loc[np.abs(df_sentiment.Score) >= avg + 1 * sigma, 'Standout'] = True

    print('Done.')

    return df_sentiment


def twitter_Sentiment(df_tweets):
    '''

    Calculates the sentiment scores and labels for a dataframe of tweets obtained with the TwitterV2API.

    :param df_tweets [pandas DataFrame]: Dataframe of tweets obtained with the TwitterV2API (as returned by get_Tweets()).
    :return df_sentiment [pandas DataFrame]: Dataframe of tweets with their sentiment scores and labels.
    '''

    df_sentiment = df_tweets.copy()

    df_sentiment['Sentiment'] = [sentiment("RoBERTa", str(x)) for x in tqdm(df_sentiment['text'])]
    df_sentiment['Score'] = [float(x[0]) for x in df_sentiment['Sentiment']]
    df_sentiment['Label'] = [str(x[1]) for x in df_sentiment['Sentiment']]

    # Remove sensitive information from DataFrame.
    df_sentiment['Cleaned Text'] = [' '.join([x if '@' not in x else '...' for x in df_sentiment['text'][i].split(' ')])
                                    for i in range(len(df_sentiment))]
    hashtags = [', '.join(df_sentiment['Cleaned Text'][i:i + 1].str.extractall(r'(\#\w+)')[0].value_counts().index) for
                i in range(len(df_sentiment))]
    # Get hashtags from tweet.
    df_sentiment['Hashtags'] = hashtags

    # Profanity Filter
    print('\nCleaning for profanity...\n')
    df_sentiment['Profanity Score'] = [profanity_Score(x) for x in tqdm(df_sentiment['Cleaned Text'])]
    profanity = np.array([df_sentiment['Profanity Score'] >= 0.5], dtype=bool)
    # Phishing Filter
    print('\nCleaning for spam/phishing attempts...\n')
    phishing = np.array([str(x) in ['www', '.com', 'http', 'https'] for x in tqdm(df_sentiment['Cleaned Text'])],
                        dtype=bool)
    mask = np.array((np.logical_not(profanity)) & (np.logical_not(phishing)), dtype=bool)[0]
    print(mask)
    # Filter: Not Profanity & Not Phishing
    df_sentiment = df_sentiment[mask].reset_index(drop=True)

    print(df_sentiment.columns)
    df_sentiment = df_sentiment.rename(columns={'text': 'Text', 'created_at': 'Publication Date'})

    print(df_sentiment.columns)
    df_sentiment = df_sentiment[[
        'Cleaned Text', 'Publication Date', 'Hashtags', 'Sentiment', 'Score', 'Label']].reset_index(
        drop=True)

    print('Finding top words...')
    df_sentiment['Top Words'] = [top_Words(x, 5) for x in tqdm(df_sentiment['Content'])]

    print('Finding standout articles...')
    sigma = np.abs(np.std(df_sentiment['Score']))
    avg = np.abs(np.mean(df_sentiment['Score']))
    k = [False] * len(df_sentiment)
    df_sentiment['Standout'] = k
    df_sentiment.loc[np.abs(df_sentiment.Score) >= avg + 1 * sigma, 'Standout'] = True

    return df_sentiment


'''
# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
'''

if __name__ == '__main__':

    message = '''
  __                 _                
 (  _  _/'_  _  _/  /_|   _ /   _ ' _ 
__)(-/)////)(-/)/  (  |/)(/((/_) /_)  
                            /            
    '''

    print(message)

    # ---------------------------------------------------------
    # User Input
    # ---------------------------------------------------------

    # News API details
    api_key = ''
    newsAPI_query = ''
    start_date = ''  # Please use format: "YYYY-MM-DD".
    end_date = ''  # Please use format: "YYYY-MM-DD".
    # Where to store the news data, change if need be
    path_news = r'C:\Users\cstevens\OneDrive - BDO South Africa\Desktop\Sentiment\_data\_newsAPI'
    path_news_sentiment = r'C:\Users\cstevens\OneDrive - BDO South Africa\Desktop\Sentiment\_data\_newsSentiment'

    # Twitter API details
    bearer_token = ''
    twitterAPI_query = ''  # Please use format: #Query
    # Where to store the twitter data, change if need be
    path_tweets = r'C:\Users\cstevens\OneDrive - BDO South Africa\Desktop\Sentiment\_data\_twitterAPI'
    path_tweet_sentiment = r'C:\Users\cstevens\OneDrive - BDO South Africa\Desktop\Sentiment\_data\_twitterSentiment'

    # ---------------------------------------------------------
    # Confirm User Input
    # ---------------------------------------------------------

    try:
        setting = int(input("Please select a setting: "
                            "\n(1) News API Data Acquisition. "
                            "\n(2) News API Data Acquisition + Sentiment Analysis, "
                            "\n(3) Twitter V2 API Data Acquisition."
                            "\n(4) Twitter V2 API Data Acquisition + Sentiment Analysis "
                            "\n(5) News API & Twitter V2 API Data Acquisition + Sentiment Analysis. "
                            "\n(6) Sentiment Analysis on an existing News API dataset."
                            "\n(7) Sentiment Analysis on an existing Twitter V2 API dataset."
                            "\nPlease select an option by typing the corresponding number: "))
    except ValueError:
        setting = None

    print(setting)
    options = [1, 2, 3, 4, 5, 6, 7]
    if (setting not in options):
        print('Invalid input.')
        sys.exit(0)

    options = [1, 2, 5]
    newsAPI_check = False  # Default
    if (setting in options):
        print('\nPlease confirm input for the News API:\n')
        print('News API Key: {}'.format(api_key))
        print('News API Query: {}'.format(newsAPI_query))
        print('News API Start Date: {}'.format(start_date))
        print('News API End Date: {}'.format(end_date))
        print('Filepath to save NewsAPI data: {}'.format(path_news))
        print('Filepath to save NewsAPI sentiment data: {}'.format(path_news_sentiment))

        if click.confirm('Is the information provided for the News API correct? Continue?', default=True):
            newsAPI_check = True
        else:
            newsAPI_check = False

    options = [3, 4, 5]
    twitterAPI_check = False  # Default
    if (setting in options):
        print('\nPlease confirm input for the Twitter V2 API:\n')
        print('Twitter V2 API Bearer Token: {}'.format(bearer_token))
        print('Twitter V2 API Query: {}'.format(twitterAPI_query))
        print('Filepath to save Twitter V2 API data: {}'.format(path_tweets))
        print('Filepath to save Twitter V2 sentiment data: {}'.format(path_tweet_sentiment))

        if click.confirm('Is the information provided for the Twitter V2 API correct? Continue?', default=True):
            twitterAPI_check = True
        else:
            twitterAPI_check = False

    # ---------------------------------------------------------
    # Retrieve Data using API Keys & Tokens
    # ---------------------------------------------------------

    if newsAPI_check == True:
        if api_key != '':
            try:
                df_news = get_News(api_key, newsAPI_query, start_date, end_date, searchIn="title")
                df_news.to_excel(r'{}\{}.xlsx'.format(path_news, newsAPI_query), index=False,
                                 header=True)
                print('{}.xlsx saved to folder: {}'.format(newsAPI_query, path_news))
            except KeyError:
                print('Could not find any News Articles. Invalid API Key or Invalid Query.')
                df_news = None
        else:
            print('No API Key set for News API.')
            df_news = None

    if twitterAPI_check == True:
        if bearer_token != '':
            try:
                df_tweets = get_Tweets(bearer_token, twitterAPI_query)
                df_tweets = df_tweets[['text', 'created_at']].rename(
                    columns={'text': 'Text', 'created_at': 'Publication Date'})
                df_tweets.to_excel(r'{}\{}.xlsx'.format(path_tweets, twitterAPI_query))
                print('{}.xlsx saved to folder: {}'.format(twitterAPI_query, path_tweets))
            except tweepy.errors.Unauthorized:
                print('Unauthorized. Bad Bearer Token.')
                df_tweets = None
            except tweepy.errors.BadRequest:
                print('Forbidden. Bad request.')
                df_tweets = None
        else:
            print('No Bearer Token set for Twitter V2 API.')
            df_tweets = None

    # ---------------------------------------------------------
    # Sentiment Analysis (News Articles)
    # ---------------------------------------------------------

    options = [2, 5]
    if (setting in options):
        if (newsAPI_check) and (df_news != None):
            print('\n\nImporting NLP models...\n\n')
            # Initialise Model
            FinBERT_model = init_Models('FinBERT')
            print('FinBERT model successfully imported.')
            # Analyse News
            print('\n\nAnalysing sentiment of News articles...\n\n')
            df_news_sentiment = news_Sentiment(df_news)
            df_news_sentiment.to_excel(r'{}\{}_sentiment.xlsx'.format(path_news_sentiment, newsAPI_query), index=False,
                                       header=True)
            print('{}_sentiment.xlsx saved to folder: {}'.format(newsAPI_query, path_news_sentiment))
        else:
            print('No News articles to analyse.')
            df_news_sentiment = None

    # ---------------------------------------------------------
    # Sentiment Analysis (Tweets)
    # ---------------------------------------------------------

    options = [4, 5]
    if (setting in options):
        if (twitterAPI_check) and (df_tweets != None):
            print('\n\nImporting NLP models...\n\n')
            # Initialise Model
            RoBERTa_model, RoBERTa_tokenizer, RoBERTa_labels = init_Models('RoBERTa')
            print('RoBERTa model successfully imported.')
            # Analyse Tweets
            print('\n\nAnalysing sentiment of Tweets...\n\n')
            df_tweets_sentiment = twitter_Sentiment(df_tweets)
            df_tweets_sentiment.to_excel(r'{}\{}_sentiment.xlsx'.format(path_tweet_sentiment, twitterAPI_query),
                                         index=False, header=True)
            print('{}_sentiment.xlsx saved to folder: {}'.format(twitterAPI_query, path_tweet_sentiment))
        else:
            print('No Tweets to analyse.')
            df_tweets_sentiment = None

    # ---------------------------------------------------------
    # Sentiment Analysis on Existing News API Dataset
    # ---------------------------------------------------------

    if (setting == 6):
        files = list_files(path_news)
        if len(files) > 0:
            print('The currently saved News API datasets are: ')
            for i in range(len(files)):
                print('{}: {}'.format(i, files[i]))

            try:
                idx = int(input("Please select the index of the file you want to perform Sentiment Analysis on."
                                "1st file: 0, 2nd file: 1, 3rd file: 2, etc...: "))
            except ValueError:
                print('Invalid input.')
                sys.exit(0)

            try:
                df_news = pd.read_excel(r'{}\{}'.format(path_news, files[idx]), index_col=False)
                print('\n\nImporting NLP models...\n\n')
                # Initialise Model
                FinBERT_model = init_Models('FinBERT')
                print('FinBERT model successfully imported.')
                # Analyse News
                print('\n\nAnalysing sentiment for News articles...\n\n')
                df_news_sentiment = news_Sentiment(df_news)
                df_news_sentiment.to_excel(
                    r'{}\{}_sentiment.xlsx'.format(path_news_sentiment, files[idx].split('.')[0]), index=False,
                    header=True)
                print('{}_sentiment.xlsx saved to folder: {}'.format(files[idx].split('.')[0], path_news_sentiment))
            except IndexError:
                print('Invalid input / Invalid index selected.')
                sys.exit(0)

        else:
            print('There are currently no saved News API datasets in {}.'.format(path_news))

    # ---------------------------------------------------------
    # Sentiment Analysis on Existing Twitter V2 API Dataset
    # ---------------------------------------------------------

    if (setting == 7):
        files = list_files(path_tweets)
        if len(files) > 0:
            print('The currently saved Twitter V2 API datasets are: ')
            for i in range(len(files)):
                print('{}: {}'.format(i, files[i]))

            try:
                idx = int(input("Please select the index of the file you want to perform Sentiment Analysis on."
                                "1st file: 0, 2nd file: 1, 3rd file: 2, etc...: "))
            except ValueError:
                print('Invalid input.')
                sys.exit(0)

            try:
                df_tweets = pd.read_excel(r'{}\{}'.format(path_tweets, files[idx]), index_col=False)
                print('\n\nImporting NLP models...\n\n')
                # Initialise Model
                RoBERTa_model, RoBERTa_tokenizer, RoBERTa_labels = init_Models('RoBERTa')
                print('RoBERTa model successfully imported.')
                # Analyse Tweets
                print('\n\nAnalysing sentiment for Tweets...\n\n')
                df_tweets_sentiment = twitter_Sentiment(df_tweets)
                df_tweets_sentiment.to_excel(
                    r'{}\{}_sentiment.xlsx'.format(path_tweet_sentiment, files[idx].split('.')[0]), index=False,
                    header=True)
                print('{}_sentiment.xlsx saved to folder: {}'.format(files[idx].split('.')[0], path_tweet_sentiment))
            except IndexError:
                print('Invalid input / Invalid index selected.')
                sys.exit(0)

        else:
            print('There are currently no saved Twitter V2 API datasets in {}.'.format(path_news))
