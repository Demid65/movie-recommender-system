# Introduction

This project aims to create a movie recommender model that would be able to recommend movies to different users based on their demographics.

# Data analysis
The [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/) was used as the primary dataset for this project, provided by the course team. Analysis of this dataset had shown that the dataset is structured like an SQL database and requires some preprocessing before a model can be trained on it. Dataset contains few corrupt entries which are to be removed on preprocessing stage, but otherwise the dataset is very usable and has valuable data. For details see [Notebook 0](../notebooks/0%20-%20Dataset%20exploration.ipynb)

# Model Implementation
This project uses 4-layer Fully-connected model with ReLU activation. It has input dimension of 43, hidden dimensions of 64, 32, 16 and output dimension of 1 (predicted rating).

# Model Advantages and Disadvantages
Created model has not the best accuracy possible, since it has little input information, but its small size allows it to be run on large databases of films which makes it practical as a recommender system for large movie database, aiming to provide its services for large audience. Its small size also allows for very fast training, allowing for easy improvement of the model when more data is collected.

# Training Process
The model was trained on 80k samples from the MovieLens dataset for 10 epochs. Remaining samples were used as training-evaluation dataset. For training process details see [Notebook 1](../notebooks/1%20-%20Classical%20model.ipynb).

# Evaluation
The model was evaluated during training on 20k samples. Mean Average Error was selected as the primary evaluation function for the model, in which it scored the result of 0.84.
The model also were re-evaluated on full dataset using evaluate.py script, with following scores:

- Mean Average Error - 0.847
- Mean Squared Error - 1.199
- Accuracy - 35% (Check out [Notebook 2](../notebooks/2%20-%20Evaluation.ipynb))

# Results
The result of this project is an extremely lightweight model for estimation of user rating for given movie, which can be ran and trained on massive datasets and can be used for recommendation on large movie databases.
