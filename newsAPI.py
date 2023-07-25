'''
# ---------------------------------------------------------
# Import Libraries
# ---------------------------------------------------------
'''


import pandas as pd
import requests

'''
# ---------------------------------------------------------
# Get Articles from News API

Full documentation here: https://newsapi.org/docs/endpoints/everything
Get API Key Here: https://newsapi.org/
# ---------------------------------------------------------
'''


def get_News(api_key, query, start_date, end_date, searchIn="title"):
    '''

    Fetch news articles with the News API, using a keyword, start date and end date.

    :param api_key: [String]; API key, purchased at: https://newsapi.org/s/google-news-api
    :param query: [String]; Keyword of API query. E.g., the name of the company/brand/person you want news about.
    :param start_date: [String]; Start Date of query. Please use format: "YYYY-MM-DD".
    :param end_date: [String]; End Date of query. Please use format: "YYYY-MM-DD".
    :param searchIn: [String]; Possible options are: "title", "description", "content". Multiple options can be selected
    by comma separation. This parameter specifies where to find the keyword/query.
    :return: df_articles [pandas DataFrame]; Dataframe containing the news articles, with: Headline, Source,
    Publication Date and URL.
    '''

    # Make API request
    url = f'https://newsapi.org/v2/everything?q={query}&from={start_date}&to={end_date}&searchIn={searchIn}' \
          f'&sortBy=popularity&apiKey={api_key}'

    response = requests.get(url)

    # Get articles
    articles = response.json()['articles']

    # Create DataFrame
    df_articles = pd.DataFrame(articles, columns=['title', 'description', 'content', 'url', 'publishedAt', 'source'])

    # Extract source id and name
    df_articles['source_id'] = df_articles['source'].apply(lambda x: x['id'] if x['id'] != None else '')
    df_articles['source_name'] = df_articles['source'].apply(lambda x: x['name'] if x['name'] != None else '')

    # Remove source column
    df_articles.drop('source', axis=1, inplace=True)

    # Print DataFrame
    df_articles.head()

    # Reformat DataFrame
    df_articles = df_articles.rename(
        columns={'title': 'Headline', 'description': 'Description', 'content': 'Content', 'url': 'URL',
                 'publishedAt': 'Publication Date', 'source_id': 'sourceID', 'source_name': 'Source'})
    df_articles = df_articles[['Headline', 'Description', 'Content', 'URL', 'Publication Date', 'Source']]

    return df_articles


'''
# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
'''

if __name__ == '__main__':
    # Where to save the dataframe / news articles. Please insert filepath here:
    path = ""

    # API Key for News API. https://newsapi.org/account
    api_key = ""
    query = "<KEYWORD>"

    # Start and end dates of query
    start_date = "YYYY-MM-DD"
    end_date = "YYYY-MM-DD"

    # Where to search. Headline "title" preferred by default.
    searchIn = "title"

    # Get news article and save dataframe as excel file.
    df_articles = get_News(api_key, query, start_date, end_date, searchIn)
    df_articles.to_excel(r'{}\{}.xlsx'.format(path, query), index=False, header=True)
