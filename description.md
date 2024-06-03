# Description of submission

We proceeded in two steps:
1. We estimate the distribution of occurence of first/second/third child using the full information about respondents and their potential children from 2007 to 2020. For this purpose, we used a recursive neural network (RNN) with gated recurrent units (GRU) and an attention mechanism to fit a time-to-event model for the probability to get a first/second/third child at each age from 18 to 45. This model is trained on the train data alone and does not use information from 2020-2023, which removes the risk of overfitting. 
This took us a lot of time to implement this model; so far we have done no parameter tuning and we still need to implement some changes, which should increase the interpretability of the model. However, the results with default parameters are encouraging: we obtain a good accuracy for predicting respondents' age at first child, which decreases for the second and third child. The precision of the model would greatly improve if we could feed it with register data, as we have relatively few people with more than 1 children in the LISS data. 
From this model, we extract a probability to get a child in the next three years after 2020 for each respondent.

2. We use the probability estimated in the neural network with other covariates to predict the outcome in an XGBoost model. The F1 score on the training set is encouraging (F1=0.7105, Accuracy=0.865), knowing that so far we have done no parameter tuning and we rely on a first rough selection of variables for both models. The probability estimated from step 1 has the highest explanatory power in the XGboost model. 

