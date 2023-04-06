import pandas as pd
import string
import spacy
from tqdm import tqdm
from constants import LIST_OF_RIGHT, LIST_OF_LEFT, LIST_OF_STOPPERS, MAPPING

# load the data
d = pd.read_csv("data/barsel_sentiment.csv", sep=";")

# calculate the number of words per sentence
d["length"] = d["sentences"].apply(lambda x: len(x.split()))
# exclude sentences of less than three words
d = d[d["length"] > 2].reset_index(drop=True)


def get_pol_orientation(string):
    if string in LIST_OF_LEFT:
        return "Left-winged"
    if string in LIST_OF_RIGHT:
        return "Right-winged"
    else:
        return None


d["Political_Oirentation"] = d["Newspaper"].apply(get_pol_orientation)

# loop through and take text before the list of stoppers
for stopper in LIST_OF_STOPPERS:
    d["article"] = d["article"].apply(lambda x: x.split(stopper)[0])
d = d.reset_index()

texts = d["article"].values

# loading in the danish news model that is used to clean data
nlp = spacy.load("da_core_news_lg")

d_tokens = d.copy()

# POS Tags to remove from tokens
removal = ["ADV", "PRON", "CCONJ", "PUNCT", "PART", "DET", "ADP", "SPACE", "NUM", "SYM"]

# Cleaning the text: leammatising the words, remove "removal", and lower text
tokens = []
for summary in tqdm(nlp.pipe(texts), total=len(texts)):
    proj_tok = [
        token.lemma_.lower()
        for token in summary
        if token.pos_ not in removal and not token.is_stop and token.is_alpha
    ]
    tokens.append(proj_tok)

# add column for tokens
d_tokens["tokens"] = tokens

# explode tokens, making each token have its own row
explosion = d_tokens["tokens"].explode().reset_index()
# create index column for merging
d_tokens = d_tokens.reset_index()

d_long = d_tokens[[x for x in d.columns if x != "tokens"]].merge(explosion)

d_long["word"] = d_long["tokens"].apply(
    lambda x: x.translate(str.maketrans("", "", string.punctuation))
)

counts = d_long["word"].value_counts().reset_index()

with open("stopwords/stopwords.txt") as f:
    stopwords_list = f.read()

stopwords = stopwords_list.split("\n")

mapping_list = list(MAPPING.keys())


def map_words(word, mapping, mapping_list):
    if word in mapping_list:
        return mapping.get(word)
    else:
        return word


d_long["word"] = d_long["word"].apply(lambda x: map_words(x, MAPPING, mapping_list))

data = d_long[~d_long["word"].isin(stopwords)]

data = data[data.columns[3:]].reset_index(drop=True)

d = d[
    [
        x
        for x in d.columns
        if x not in ["index", "Unnamed: 0", "merge_index", "X", "length"]
    ]
]

data = data[
    [
        x
        for x in data.columns
        if x not in ["index", "Unnamed: 0", "merge_index", "X", "length"]
    ]
]

# add political information for  article sentiment and clean for sentences with less than 3 words
d.to_csv("data/barsel_data.csv", index=False)

# final data for LDA
data.to_csv("data/preprocessed_words.csv", index=False)
