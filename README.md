# English COVID-19 Fake News Detection

This repository contains the code for the [Constraint@AAAI2021 - COVID19 Fake News Detection in English](https://competitions.codalab.org/competitions/26655) 
challenge that was conducted during December 2020. This submission was made based on the training of several transformer based models.

The submission scored an F1 score of 0.96 on the test dataset using the transfer learning method on [RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html) model. 
We performed an extensive research on the dataset and drove insights based on the hashtags and twitter handles that were part of the tweets and generated 
correlation graphs before training the model. This research helped us in performing an extensive pre-processing which improved the overall score of our model.
We put together all of our pre-processing scripts under a new python package (also released on pip) for consumption by the open source community. 
Check out the package here on github: [PREVIS](https://github.com/zutshianand/Previs) and [this link](https://pypi.org/project/previs/) on installation instructions.

In addition to the provided dataset in the Codalab competition, we aggregated COVID-19 tweets dataset from other sources as well and then again 
trained our model which helped it further improve the overall score on the validation and test dataset.
