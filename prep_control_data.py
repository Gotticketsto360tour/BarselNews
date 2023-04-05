import pandas as pd
import seaborn as sns

# political = pd.read_csv("data/Average_New2023.csv", sep=";")

# political = pd.read_csv("data/data_political.csv", sep=";")

# political = pd.read_csv("data/data_political_april.csv", sep=";")

controls = pd.read_csv("data/Average_Article_sentiment.csv", sep=";")

list_of_right = [
    "Jyllands-Posten",
    "BT.dk",
    "B.T.",
    "Børsen",
    "BørsenLørdag",
    "Weekendavisen.dk",
    "JPAarhus",
    "B.dk",
    "Berlingske",
    "BørsenSøndag",
    "Jyllands-Posten",
    "Weekendavisen",
]
list_of_left = [
    "Information.dk(Abonnementsområde)",
    "Information.dk",
    "Information",
    "EkstraBladet",
    "Politiken",
]


def get_pol_orientation(string):
    if string in list_of_left:
        return "Left-winged"
    if string in list_of_right:
        return "Right-winged"
    else:
        return None


# political["political_orientation"] = political["Newspaper"].apply(get_pol_orientation)

# political = political[[x for x in political.columns if x != "Unnamed: 0"]]

# political.to_csv("data/control_data.csv", sep=";", index=False)

controls["Political_Orientation"] = controls["Newspaper"].apply(get_pol_orientation)

# controls = controls[controls["Political_Orientation"].notna()].reset_index(drop=True)
# controls = controls[~controls["køn"].isin(["?", "Begge"])].reset_index(drop=True)

controls["length"] = controls["sentences"].apply(lambda x: len(x.split()))
controls = controls[controls["length"] > 2].reset_index(drop=True)
controls = controls[[x for x in controls.columns if x != "Unnamed: 0"]]

controls.to_csv("data/control_data.csv", index=False)
