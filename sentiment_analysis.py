# import the different packages

import pandas as pd
from danlp.models import load_bert_tone_model
from tqdm import tqdm

data = pd.read_csv("data/average_article_sentences_.csv", sep=",")

# load the tone model and call it classifier
classifier = load_bert_tone_model()

# get the comments
texts = data["sentences"].values

# SENTIMENT ANALYSIS
# _______________

# make stupid lists to store the data
positive_prob = []
neutral_prob = []
negative_prob = []
objective_prob = []
subjective_prob = []

for comment in tqdm(texts):
    # get the prediction probabilities for each comment
    probs = classifier.predict_proba(comment)
    # split the probabilities into whether it's sentiment of objective
    sentiment, angle = probs[0], probs[1]

    # store the data
    positive_prob.append(sentiment[0])
    neutral_prob.append(sentiment[1])
    negative_prob.append(sentiment[2])
    objective_prob.append(angle[0])
    subjective_prob.append(angle[1])

# add the data to the data frame
(
    data["positive_prob"],
    data["neutral_prob"],
    data["negative_prob"],
    data["objective_prob"],
    data["subjective_prob"],
) = (positive_prob, neutral_prob, negative_prob, objective_prob, subjective_prob)

# store the csv
data.to_csv("data/average_article_sentences_probs.csv", sep=";", index=False)
