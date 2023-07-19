'''
# ---------------------------------------------------------
# Import Libraries
# ---------------------------------------------------------
'''

from transformers import AutoModelForSequenceClassification
from finbert.finbert import *
from utils import *
from scipy.special import softmax
import logging
logging.disable(logging.INFO)


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
    sentiment_label = str(val[idx])

    return sentiment_score, sentiment_label


def RoBERTa():
    '''

    Create a pre-trained RoBERTa (A Robustly Optimized BERT Pretraining Approach) model
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
    sentiment_label = labels[np.argmax(scores)]

    return sentiment_score, sentiment_label


'''
# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
'''

if __name__ == '__main__':
    text = "This is truly horrible."
    model, tokenizer, labels = RoBERTa()
    score, label = RoBERTa_Sentiment(model, tokenizer, labels, text)
    print(score, label)

    text = "M&T Bank beats profit estimates as higher rates boost interest income."
    model = FinBERT()
    score, label = FinBERT_Sentiment(model, text)
    print(score, label)
