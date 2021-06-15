import pandas as pd

dataframe = pd.read_csv("grb_table.txt", sep="\t")

dataframe.to_csv("grb_table.csv", index=None)
