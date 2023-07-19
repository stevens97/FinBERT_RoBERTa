'''
# ---------------------------------------------------------
# Import Libraries
# ---------------------------------------------------------
'''

from urllib.request import urlopen
from bs4 import BeautifulSoup

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def get_full_content(URL):
    '''

    :param URL:
    :return:
    '''

    html = urlopen(URL).read()
    soup = BeautifulSoup(html, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()  # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    full_text = '\n'.join(chunk for chunk in chunks if chunk)

    return full_text


def preprocess_text(text):
    '''

    :param text:
    :return:
    '''

    def remove_emoji(text):
        '''

        :param text:
        :return:
        '''
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    tokens = [remove_emoji(token) for token in lemmatized_tokens]
    # Join the tokens back into a string
    processed_text = ' '.join(tokens)

    return processed_text
