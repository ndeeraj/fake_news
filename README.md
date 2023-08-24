# Fake news detection

We have followed two approaches for building models to detect fake news.

## Approach 1: Feature engineering

- featurize_data.ipynb: generates features for the dataset.
- models_logr_svm_rf.ipynb: trains logistic regression, SVMs, Random forest models on the features.
- model_feedforward.ipynb: trains a feedforward neural network on the features.

## Approach 2: Using DistilBERT for classification

- src/fake_news/bert/train.py: trains the pretrained transformer on the classification task.​

steps: 1000 </br>
eval_loss: 0.03982425853610039 </br>
eval_precision: 1.0 </br>
eval_recall: 0.9885155253083794 </br>
eval_f1: 0.9942245989304812 </br>
