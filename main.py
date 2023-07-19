'''
# ---------------------------------------------------------
# Import Libraries
# ---------------------------------------------------------
'''

from transformers import AutoModelForSequenceClassification
from finbert.finbert import *
from utils import *


def finbert():
    finbert = "ProsusAI/finbert"
    model = AutoModelForSequenceClassification.from_pretrained(finbert)
    return model


def finbert_Score(model, text):
    result_df = pd.DataFrame(predict(text, model))

    print(result_df['sentence'])
    print(result_df['logit'])
    print(result_df['prediction'])
    print(result_df['sentiment_score'])

    return None


def roberta():
    # load model and tokenizer
    # For Twitter Data
    roberta = "cardiffnlp/twitter-roberta-base-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)
    labels = ['Negative', 'Neutral', 'Positive']

    return model, tokenizer, labels


def roberta_Score(model, tokenizer, labels, text):
    text = preprocess_text(text)

    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
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

    sentiment_score = (negativity * -1) + (neutrality * 0) + (positivity * 1)
    sentiment_label = labels[np.argmax(scores)]

    return sentiment_score, sentiment_label


'''
# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
'''

if __name__ == '__main__':
    print()
