# import the different packages

import pandas as pd
from danlp.models import load_bert_tone_model
from tqdm import tqdm

barsel_data = pd.read_csv("data/barsel_sentences.csv", sep=",")
control_data = pd.read_csv("data/control_sentences.csv", sep=",")

# load the tone model
classifier = load_bert_tone_model()

# SENTIMENT ANALYSIS
# _______________


def sentiment_analysis(comment: str):
    probs = classifier.predict_proba(comment)
    sentiment, angle = probs[0], probs[1]
    return (sentiment[0], sentiment[1], sentiment[2], angle[0], angle[1])


for data in [barsel_data, control_data]:
    (
        data["positive_prob"],
        data["neutral_prob"],
        data["negative_prob"],
        data["objective_prob"],
        data["subjective_prob"],
    ) = zip(*data["sentences"].apply(sentiment_analysis))

barsel_data.to_csv("data/barsel_sentiment.csv", sep=";", index=False)
control_data.to_csv("data/control_sentiment.csv", sep=";", index=False)
