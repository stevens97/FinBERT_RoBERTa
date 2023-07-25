'''
# ---------------------------------------------------------
# Import Libraries
# ---------------------------------------------------------
'''

import os
import urllib
from urllib.request import urlopen
from bs4 import BeautifulSoup
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import time
from threading import Thread
from detoxify import Detoxify


def profanity_Score(text):
    '''

    Calculates the profanity score (profanity / obscenity / hate speech) for a given text.
    Assigns a score between 0 and 1 for each category.

    Returns the worst score to filter for profanity.

    Original Model: https://huggingface.co/unitary/toxic-bert
    Original Source Code: https://github.com/unitaryai/detoxify
    Original Papers: https://arxiv.org/abs/1703.04009; https://arxiv.org/abs/1905.12516

    :param text: [String]; Text to analyse.
    :return: Worst score.
    '''
    return float(np.array(list(Detoxify('original').predict(text).values())).max())


def list_files(dir_path):
    '''

    List files stord in directory.

    :param dir_path: [String]; Filepath / directory.
    :return: res: [String array]; List of all files.
    '''
    # list to store files
    res = []
    try:
        for file_path in os.listdir(dir_path):
            if os.path.isfile(os.path.join(dir_path, file_path)):
                res.append(file_path)
    except FileNotFoundError:
        print(f"The directory {dir_path} does not exist")
    except PermissionError:
        print(f"Permission denied to access the directory {dir_path}")
    except OSError as e:
        print(f"An OS error occurred: {e}")
    return res


def preprocess_text(text):
    '''

    Pre-processing text for NLP tasks.

    Pre-processing involves: (1) Text clean-up. (2) Tokenisation. (3) Lemmatisation.

    :param text: [String]; Text to pre-process.
    :return: processed_text [String]; Cleaned text.
    '''

    def clean(text):
        '''

        Remove unwanted symbols from text.

        :param text: [String]; Text to process.
        :return: cleaned_text [String]; Cleaned text.
        '''
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   "]+", flags=re.UNICODE)
        cleaned_text = emoji_pattern.sub(r'', text)

        return cleaned_text

    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    tokens = [clean(token) for token in lemmatized_tokens]
    # Join the tokens back into a string
    processed_text = ' '.join(tokens)

    return processed_text


def get_full_content(URL):
    '''

    Scrape full-text from website given its URL.

    NOTE: Always be careful when scraping content from websites and be sure not to violate any privacy rights.

    :param URL: [String]; URL of the website.
    :return: full_text [String]; Full text from the website.
    '''

    try:
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
    except urllib.error.HTTPError:
        print('HTTPError')
        full_text = None
    except urllib.error.URLError:
        print('URLError')
        full_text = None

    return full_text


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


def call(f, *args, timeout=5, **kwargs):
    i = 0
    t = ThreadWithReturnValue(target=f, args=args, kwargs=kwargs)
    t.daemon = True
    t.start()
    while True:
        if not t.is_alive():
            break
        if timeout == i:
            print("Connection Timeout")
            return
        time.sleep(1)
        i += 1
    return t.join()


def fetch_Articles(write_path, df_news):
    '''

    Scrape all URLs in a given dataframe, as originally obtained from the News API.

    :param write_path: [String]; Directory to save the files in.
    :param df_news: [pandas DataFrame]; Dataframe as obtained with the News API
    :return: None;
    '''
    for i in range(len(df_news)):
        print(df_news['URL'][i])
        print(i)

        URL = df_news['URL'][i].replace('/', '_').replace('\\', '_').replace(':', '_')

        file = r'{}.txt'.format(URL)
        file_path = r'{}/{}'.format(write_path, file)

        if os.path.exists(file_path) == False:
            temp = call(get_full_content, timeout=10, URL=df_news['URL'][i])
            print(temp)

    return None
