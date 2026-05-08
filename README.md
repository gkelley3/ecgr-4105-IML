# ecgr-4105-IML
Coursework related to ECGR 4105 Introduction to Machine Learning for progress towards B.S.E.E. at UNC Charlotte

Brief description of each .py script:

dataset_generator.py - generates synthetic 5,000-sample dataset using discrete grid search. Random system parameters and initial conditions are the input labels, and the discretized local optima controller gains are the output labels.

analyze_dataset.py - produces a correlation matrix and several plots for features within the dataset.

train-model.py - trains a model using MLP regressor for energy shaping gain k_E and random forest for LQR gains k_1 and k_2.

evaluate_controller_performance.py - uses the model to predict controller gains in a closed-loop simulation of the inverted pendulum system using randomized system parameters and initial conditions. Includes quantitative and representative data comparing the ML-predicted controller to the discrete grid search controllers from the dataset.  

Please see the following [Google Drive folder](https://drive.google.com/drive/folders/1KTLUxKmWK50N2itjYOGSJO-8P9EAw2wW?usp=drive_link) for the trained model (.joblib), the presentation slide deck, and an example video of the swing-up inverted pendulum system.
