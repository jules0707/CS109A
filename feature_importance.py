# %% [markdown]
# ## Title :
# 
# Exercise: Feature Importance
# 
# The goal of this exercise is to compare two feature importance methods; MDI, and Permutation Importance. For a discussion on the merits of each go to this <a href="https://scikit-learn.org/stable/modules/permutation_importance.html" target="_blank">link</a>.
# 
# ## Description :
# 
# <img src="./fig/fig2.png" style="width: 1000px;">
# 
# ## Instructions:
# 
# - Read the dataset `heart.csv` as a pandas dataframe, and take a quick look at the data.
# - Assign the predictor and response variables as per the instructions given in the scaffold.
# - Set a max_depth value.
# - Define a `DecisionTreeClassifier` and fit on the entire data.
# - Define a `RandomForestClassifier` and fit on the entire data.
# - Calculate Permutation Importance for each of the two models. Remember that the MDI is automatically computed by sklearn when you call the classifiers.
# - Use the routines provided to display the feature importance of bar plots. The plots will look similar to the one given above.
# 
# ## Hints: 
# 
# <a href="https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#" target="_blank">forest.feature_importances_</a>
# Calculate the impurity-based feature importance.
# 
# <a href="https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance" target="_blank">sklearn.inspection.permutation_importance()</a>
# Calculate the permutation-based feature importance.
# 
# <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html" target="_blank">sklearn.RandomForestClassifier()</a>
# Returns a random forest classifier object.
# 
# <a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html" target="_blank">sklearn.DecisionTreeClassifier()</a>
# Returns a decision tree classifier object.
# 
# 
# **NOTE** - MDI is automatically computed by sklearn by calling RandomForestClassifier and/or DecisionTreeClassifier.

# %%
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import helper
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from helper import plot_permute_importance, plot_feature_importance


# %%
# Read the dataset "heart.csv"
df = pd.read_csv("heart.csv")

# Take a quick look at the data 
df.head()


# %%
# Assign the predictor and response variables.
# 'AHD' is the response and all the other columns are the predictors
X = df.drop('AHD', axis=1)
X_design = pd.get_dummies(X, drop_first=True)
y = df['AHD']


# %%
# Set the model parameters

# The random state is fized for testing purposes
random_state = 44

# Choose a `max_depth` for your trees 
max_depth = 3


# %% [markdown]
# ### SINGLE TREE

# %%
### edTest(test_decision_tree) ###

# Define a Decision Tree classifier with random_state as the above defined variable
# Set the maximum depth to be max_depth
tree = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)

# Fit the model on the entire data
tree.fit(X_design, y)

# Using Permutation Importance to get the importance of features for the Decision Tree 
# with random_state as the above defined variable
tree_result = permutation_importance(tree, X_design, y, random_state=random_state)


# %% [markdown]
# ### RANDOM FOREST

# %%
### edTest(test_random_forest) ###

# Define a Random Forest classifier with random_state as the above defined variable
# Set the maximum depth to be max_depth and use 10 estimators
forest = RandomForestClassifier(random_state=random_state, n_estimators=10, max_depth=max_depth)

# Fit the model on the entire data
forest.fit(X_design, y)

# Use Permutation Importance to get the importance of features for the Random Forest model 
# with random_state as the above defined variable
forest_result = permutation_importance(forest, X_design, y, random_state=random_state)


# %% [markdown]
# ### PLOTTING THE FEATURE RANKING

# %%
# Helper code to visualize the feature importance using 'MDI'
plot_feature_importance(tree,forest,X_design,y)

# Helper code to visualize the feature importance using 'permutation feature importance'
plot_permute_importance(tree_result,forest_result,X_design,y)


# %% [markdown]
# ⏸ A common criticism for the MDI method is that it assigns a lot of importance to noisy features (more here). Did you make such an observation in the plots above?

# %%
### edTest(test_chow1) ###
# Type your answer within in the quotes given
answer1 = 'yes'


# %% [markdown]
# ⏸ After marking, change the max_depth for your classifiers to a very low value such as 
# 3
# , and see if you see a change in the relative importance of predictors.

# %%
### edTest(test_chow2) ###
# Type your answer within in the quotes given
answer2 = 'yes'



