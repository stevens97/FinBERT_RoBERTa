'''
# ---------------------------------------------------------
# Import Libraries
# ---------------------------------------------------------
'''

import tweepy
import numpy as np
import pandas as pd
import ast

'''
# ---------------------------------------------------------
# Get Tweets from Twitter V2 API

Full documentation here: https://developer.twitter.com/en/docs/twitter-api/data-dictionary/object-model/tweet
Get API Key Here: https://developer.twitter.com/en/products/twitter-api

If you don't understand search queries, there is an excellent introduction to it here: 
https://github.com/twitterdev/getting-started-with-the-twitter-api-v2-for-academic-research/blob/main/modules/5-how-to-write-search-queries.md
# ---------------------------------------------------------
'''


def get_Tweets(bearer_token, query):
    '''

    Fetch most recent tweets with the Twitter V2 API, using a hashtag to look for.

    :param bearer_token: [String]; Bearer token as purchased from the Twitter V2 API.
    :param query: [String]; Hashtag to look for. Please include hashtag as part of string. E.g., #EXAMPLE
    :return: df_tweets [pandas DataFrame]; Dataframe containing the tweets and all meta-data.
    '''

    client = tweepy.Client(bearer_token=bearer_token)

    # Get tweets that contain the hashtag
    # -is:retweet means I don't wantretweets
    # lang:en is asking for the tweets to be in english
    query = '{} -is:retweet lang:en'.format(query)
    tweets = tweepy.Paginator(client.search_recent_tweets, query=query,
                              tweet_fields=['context_annotations', 'created_at'], max_results=100).flatten(limit=1000)

    # --------------------
    # Initialise Variables
    # --------------------
    ids = []
    text = []
    edit_history_ids = []
    attachments = []
    author_id = []
    context_annotations = []
    conversation_id = []
    created_at = []
    edit_controls = []
    entities = []
    in_reply_to_user_id = []
    lang = []
    non_public_metrics = []
    organic_metrics = []
    possibly_sensitive = []
    promoted_metrics = []
    public_metrics = []
    referenced_tweets = []
    reply_settings = []
    source = []
    withheld = []

    # Get Tweet info
    for tweet in tweets:
        # tweet ID
        ids = np.append(ids, tweet.id)
        # text
        text = np.append(text, tweet.text)
        edit_history_ids = np.append(edit_history_ids, tweet.edit_history_tweet_ids)
        in_reply_to_user_id = np.append(in_reply_to_user_id, tweet.in_reply_to_user_id)
        # attachments
        attachments = np.append(attachments, tweet.attachments)
        # author ID and source
        author_id = np.append(author_id, tweet.author_id)
        source = np.append(source, tweet.source)
        # conversation ID
        conversation_id = np.append(conversation_id, tweet.conversation_id)
        # Context
        if (len(str(tweet.context_annotations)) > 0):
            context_annotations = np.append(context_annotations, str(tweet.context_annotations))
        else:
            context_annotations = np.append(context_annotations, 'None')
        # Location
        created_at = np.append(created_at, tweet.created_at)
        # metrics
        non_public_metrics = np.append(non_public_metrics, tweet.non_public_metrics)
        organic_metrics = np.append(organic_metrics, tweet.organic_metrics)
        possibly_sensitive = np.append(possibly_sensitive, tweet.possibly_sensitive)
        promoted_metrics = np.append(promoted_metrics, tweet.promoted_metrics)
        # other
        edit_controls = np.append(edit_controls, tweet.edit_controls)
        entities = np.append(entities, tweet.entities)
        lang = np.append(lang, tweet.lang)
        referenced_tweets = np.append(referenced_tweets, tweet.referenced_tweets)
        reply_settings = np.append(reply_settings, tweet.reply_settings)
        withheld = np.append(withheld, tweet.withheld)

    # --------------------
    # Data Post-processing
    # --------------------
    context = [ast.literal_eval(str(x)) for x in context_annotations]
    created_at = [x.strftime("%Y-%m-%d %H:%M:%S") for x in created_at]

    # --------------------
    # Save Results
    # --------------------

    details = {'id': ids, 'text': text, 'edit_history_ids': edit_history_ids,
               'in_reply_to_user_id': in_reply_to_user_id, 'attachments': attachments,
               'author_id': author_id, 'source': source, 'conversation_id': conversation_id, 'context': context,
               'created_at': created_at, 'non_public_metrics': non_public_metrics,
               'organic_metrics': organic_metrics, 'possibly_sensitive': possibly_sensitive,
               'promoted_metrics': promoted_metrics, 'edit_controls': edit_controls, 'entities': entities, 'lang': lang,
               'referenced_tweets': referenced_tweets,
               'reply_settings': reply_settings, 'withheld': withheld}

    df_tweets = pd.DataFrame(details)

    return df_tweets


'''
# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
'''

if __name__ == '__main__':
    # Where to save the dataframe / news articles. Please insert filepath here:
    path = ""

    # Bearer Token for the Twitter V2 API. https://developer.twitter.com/en/products/twitter-api
    bearer_token = ""
    # Query / Hashtag to look for.
    query = "#EXAMPLE"

    df_tweets = get_Tweets(bearer_token, query)
    df_tweets.to_excel(r'{}\{}.xlsx'.format(path, query), index=False, header=True)
