import pandas as pd
from constants import LIST_OF_LEFT, LIST_OF_RIGHT

controls = pd.read_csv("data/control_sentiment.csv", sep=";")


def get_pol_orientation(string):
    if string in LIST_OF_LEFT:
        return "Left-winged"
    if string in LIST_OF_RIGHT:
        return "Right-winged"
    else:
        return None


controls["Political_Orientation"] = controls["Newspaper"].apply(get_pol_orientation)

controls["length"] = controls["sentences"].apply(lambda x: len(x.split()))
controls = controls[controls["length"] > 2].reset_index(drop=True)
controls = controls[
    [
        x
        for x in controls.columns
        if x not in ["Unnamed: 0", "merge_index", "X", "length"]
    ]
]

controls.to_csv("data/control_data.csv", index=False)
