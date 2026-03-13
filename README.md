# Trajectory Prediction for Autonomous Driving: RNN and Transformer Approaches

Trajectory prediction is a fundamental problem in autonomous driving. An autonomous vehicle must anticipate how surrounding agents such as cars, cyclists, and pedestrians will move in the near future in order to plan safe and efficient driving strategies. The goal of trajectory prediction is to forecast the future motion of road users based on their observed past trajectories.

In this project, we study the problem of predicting the future two-dimensional trajectories of road agents using LSTMs, GRUs and Transformers. Given the past motion of a target agent, the task is to predict its future positions over a fixed prediction horizon.

We have developed a central pre-processing strategy, 
which standardizes the input data for all models. This includes normalizing agent positions, handling missing data, and segmenting trajectories into fixed-length sequences. By maintaining a consistent preprocessing pipeline, we ensure fair comparisons between models and reproducibility of results. 

Each model has been given its own seperate branch to allow for independent implementation and experimentation.

## Baseline Model
As a reference method, we implement a [constant velocity baseline model](https://github.com/ploewen/Trajectory_Prediction/tree/yongxin-baseline) .

## RNN Model
The [RNN model](https://github.com/ploewen/Trajectory_Prediction/tree/rnn-model) leverages sequential neural networks such as LSTMs and GRUs to capture temporal dependencies in agent trajectories for accurate future position prediction.

## Transformer Model
The [Transformer model](https://github.com/ploewen/Trajectory_Prediction/tree/transformer-model) utilizes self-attention mechanisms to model complex interactions and long-range dependencies in trajectory data for improved prediction accuracy.