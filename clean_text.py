import pandas as pd
import string
import spacy
from tqdm import tqdm

# load the data
d = pd.read_csv("data/Article_sentiment.csv", sep=";")
d_pol = pd.read_csv("data/data_political1.csv")

# calculate the number of words per sentence
d["length"] = d["sentences"].apply(lambda x: len(x.split()))
# exclude sentences of less than three words
d = d[d["length"] > 2].reset_index(drop=True)
# merge with political data to include political orientation for articles
political = d_pol[["ID", "Political_Orientation"]]
d = d.merge(political, on="ID")

# list of endings of articles that need to excluded from the texts
list_of_stoppers = [
    "LINEA SØGAARD-LIDELL EU-parlamentariker, Venstre.",
    " I LLU ST RATION: ",
    " Kilde: Beskæftigelsesministeriet. ,",
    " dorte.boddum@finans.dk",
    " mie.l.raatz@jp.dk ",
    " segj@information.dk",
    ". (arkivfoto)",
    " FOTO: ",
    " frederikke.traeholt@jp.dk",
    ". Ritzau.",
    "Camilla Gregersen Dansk Magisterforening.",
    " makr@information.dk",
    "MICHELLA MEIER-MORSI. , Michella Meier-Morsi og Mark Morsi med deres fem børn. Privatfoto",
    " (ARKIV)",
    " loka@berlingske.dk , ",
    " jonas.proschold@pol.dk",
    "SIDE 8-9. FOTO: ",
    " elisabet.svane@pol.dk ,",
    " ILLUSTRATION: ",
    "Kilder: EU-",
    "Kilder: Folketingets",
    "Kilde: \\ ",
    "Kilder: Danmarks Statistik, Politiken, Ritzau. ",
    " sonne@information.dk ",
    " , laura.nissen@jp.dk",
    " PR-foto",
    "Ark kivfoto: ",
    "anders.m.bruun@finans.dk Illustration: ",
    "winther@k.dk Fakta:",
    "FOTO: PER",
    "Fakta: FAKTA ",
    "Fejl og Fakta 28.10.2021:",
    "Fakta: Hvad mener du? Send ",
    " ELSE JOHANNESSEN,",
    " Fakta: BLÅ BOG ",
    "Se alle 140 underskrivere på information.dk/ deltag.",
    " (Foto: ",
    " (© COLORBOX) ",
    "hejs@berlingske.dk tbre@berlingske.dk",
    " KILDER: ",
    " information. dk/deltag ",
    " Arkivfoto: ",
    "Se klippet fra samtalen mellem Alex Vanopslagh og Søren Pind her.",
    "LÆS OGSÅ KRONIKKEN SIDE ",
    "PRIVAT",
    " Berlingske Grafik: ",
    "/ ritzau/",
    " Læs mere her. , FOLD UD ",
    "Alt mediemateriale fra Infomedia er ophavsretligt beskyttet.",
    "/Ritzau/",
    " Fold sammen Læs mere ",
    "/ritzau/",
    " Foto: ",
]
# loop through and take text before the list of stoppers
for stopper in list_of_stoppers:
    d["article"] = d["article"].apply(lambda x: x.split(stopper)[0])
d = d.reset_index()

texts = d["article"].values

# loading in the danish news model that is used to clean data
nlp = spacy.load("da_core_news_lg")


# drop duplicates to have a row per article
# d_tokens = d.drop_duplicates("merge_index").reset_index(drop=True)
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

mapping = {
    "socialdemokratisk": "socialdemokrati",
    "sammenligning": "sammenligne",
    "regle": "regler",
    "parret": "par",
    "orlovsug": "orlov",
    "økonom": "økonomi",
    "økonomisk": "økonomi",
    "økonomii": "økonomi",
    "økonomiisk": "økonomi",
    "mødre": "mor",
    "moder": "mor",
    "mødr": "mor",
    "mødrene": "mor",
    "mødree": "mor",
    "mødree": "mor",
    "mødreene": "mor",
    "modell": "model",
    "mænds": "mand",
    "mændene": "mand",
    "kvinder": "kvinde",
    "kvinders": "kvinde",
    "kvinderne": "kvinde",
    "kvind": "kvinde",
    "kvinderr": "kvinde",
    "kvinderrelig": "kvinde",
    "kvinderrer": "kvinde",
    "kvinderererererre": "kvinde",
    "kvindererererer": "kvinde",
    "konsekvense": "konsekvens",
    "konkre": "konkret",
    "idé": "ide",
    "forvej": "forvejen",
    "forældr": "forældre",
    "famili": "familie",
    "familiee": "familie",
    "fagbevægelsen": "fagbevægelse",
    "fagbevægels": "fagbevægelse",
    "fagbevægelsenens": "fagbevægelse",
    "fars": "far",
    "fædr": "far",
    "fædrene": "far",
    "fædrenes": "far",
    "fædres": "far",
    "fader": "far",
    "farene": "far",
    "farenes": "far",
    "fares": "far",
    "barselsug": "barselsuge",
    "barsle": "barsel",
    "barslenn": "barsel",
    "barslen": "barsel",
    "barselslovgivning": "barselslov",
    "barselsorloven": "barselsorlov",
    "barselsregler": "barselsregel",
    "barselsregle": "barselsregel",
    "barselsreglerrr": "barselsregel",
    "barselsreglerr": "barselsregel",
    "barselsugeee": "barselsuge",
    "barselsugee": "barselsuge",
    "børn": "barn",
    "børnenes": "barn",
    "børns": "barn",
    "forældree": "forældre",
    "konkrett": "konkret",
    "ligestillingsminist": "ligestillingsminister",
    "ligestillingsordføre": "ligestillingsordfører",
    "fagbevægelsenenenenen": "fagbevægelse",
}

with open("stopwords/stopwords.txt") as f:
    stopwords_list = f.read()

stopwords = stopwords_list.split("\n")

mapping_list = list(mapping.keys())


def map_words(word, mapping, mapping_list):
    if word in mapping_list:
        return mapping.get(word)
    else:
        return word


d_long["word"] = d_long["word"].apply(lambda x: map_words(x, mapping, mapping_list))

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
