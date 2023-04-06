# INSERT TITLE OF THE PAPER
This repository contains the code for reproducing the results of the analysis of INSERT TITLE OF THE PAPER. 
The data from the articles is not available for sharing due to restrictions on data sharing from Infomedia. With access to Infomedia, this repository aims to illuminate the specific preprocessing and analysis decisions made, and to make it reproducible for interested parties.

## Steps to reproduce the analysis
1. Follow the instructions and code in *data_cleaning/Articles_to_csv.Rmd*. This converts the text in the PDFs of articles and converts them to a .csv. 
2. Run *sentiment_analysis.py* to use the BERT_Tone model to classify sentences as either negative, neutral or positive.
3. 