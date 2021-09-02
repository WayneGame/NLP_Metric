# Efficient Metrics
## Goals  
For our efficient metrics experiments we use the metrics 
- BERTScore  
- MoverScore
- XMoverScore
- SuperT

and replace the utilized BERT model with better performing and more efficient variants. The exact thoughts and approaches can be found in [our report](https://github.com/WayneGame/NLP_Metric/blob/main/MSE_Efficient_Metrics_v3.pdf).

## Overview
[BERTScore](https://github.com/WayneGame/NLP_Metric/tree/main/BERTScore), [MoverScore](https://github.com/WayneGame/NLP_Metric/tree/main/MoverScore), [XMoverScore](https://github.com/WayneGame/NLP_Metric/tree/main/XMoverScore) and [SuperT](https://github.com/WayneGame/NLP_Metric/blob/main/SUPERT.ipynb) can be found in their respective directories along with our received results. For KoBe and COMET it was not possible for us to get expressive results.

## Usage

### BERTScore
To experiment with BERTScore, go into its directory and open the [BERTScore.ipynb](https://github.com/WayneGame/NLP_Metric/blob/main/BERTScore/BERTScore.ipynb)
- To execute the file you need Google Drive and need to put the [scores file](https://github.com/WayneGame/NLP_Metric/blob/main/scores_deen.csv) into the specified directory from the variable ``` dpath``` (```/content/drive/MyDrive/data/MSE/scores_deen.csv```)
- Upon executing the script will automatically iterate over the given variants and calculate their scores, time and memory usage and save a CSV file containing the results

### MoverScore
To experiment with MoverScore, go into its directory and open the [MoverScore_2_0.ipynb](https://github.com/WayneGame/NLP_Metric/blob/main/MoverScore/MoverScore_2_0.ipynb)
- To execute the file you need Google Drive and need to put the [scores file](https://github.com/WayneGame/NLP_Metric/blob/main/scores_deen.csv) into the specified directory from the variable ``` dpath``` (```/content/drive/MyDrive/data/MSE/scores_deen.csv```)
- Before executing, you need to specify the ```model_name``` so it can be used as an environment variable by ```moverscore_v2```
- Then the results of this variant will be saved as a CSV file
- when running the script multiple times with different  ```model_name```s, all resulting files can be merged with [merge_scores](https://github.com/WayneGame/NLP_Metric/blob/main/MoverScore/merge_scores.ipynb), which will save one resulting CSV file comparing the variants

### XMoverScore
To experiment with XMoverScore and play around with different models, go into its directory and open [XMoverScore/demo.py](https://github.com/WayneGame/NLP_Metric/blob/main/XMoverScore/demo.py), where you find a modified script from the source folder.
- The attribute ```variants``` is a list with all transformers you want to test. If its a local model, you need to specify the source folder in the [XMoverScore/scorer.py](https://github.com/WayneGame/NLP_Metric/blob/main/XMoverScore/scorer.py) - script. If the transformer is on``` huggingface.co```, an additional adjustement in [XMoverScore/scorer.py](https://github.com/WayneGame/NLP_Metric/blob/main/XMoverScore/scorer.py) could be necessary.
- The results will be saved as a CSV file in the same folder.
- Plotting: To get the same plots from our report the [XMoverScore/Datenaufbereitung.ipynb](https://github.com/WayneGame/NLP_Metric/blob/main/XMoverScore/Datenaufbereitung.ipynb) -notebook can be used. The values from the generated CSV-file will be used.

### SUPERT
To experiment with SUPERT, open [SUPERT.ipynb](https://github.com/WayneGame/NLP_Metric/blob/main/SUPERT.ipynb). There should be the code to execute all BERT variations with SUPERT

To get the other BERT variations running with the given SUPERT-Notebook:

1) execute the cells, where the BERT-Models are downloaded
2) Go to the cached download directory of the BERT models (on mac os: ```~/.cache/torch/sentence_transformers/```) and open the folder of the downloaded models
3) Open ```modules.json``` and in line 6 rename ```....models.Transformer"``` to ```....models.DistilBERT"``` exemplary for DistilBERT (Write the correct model name depending of the BERT variation) 

The correct naming of the model can be taken from the [class names](https://github.com/WayneGame/NLP_Metric/tree/main/SUPERT/sentence_transformers/models).

