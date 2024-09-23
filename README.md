This is a transcript of a jupyter notebook changing the code blocks into functions.

The Approach

I will split the data into training and test sets using an 80 - 20 split. I will then train
and test the data on three different machine learning algorithms with the goal to see
which of the three algorithms is best for solving the problem. For each algorithm I will
tune the most important hyperparameters utilising grid search, this will likely need to
use parallelization due to the amount of hyperparameters to tune and the amount of
options to set them at. For this i aim to use dask to utilise all of my cpu cores.
II will then run the algorithms with the tuned hyperparameters on the test set utilising
confusion matrices and accuracy scores to score them. This should tell me which
algorithm was the best for this problem.

Random Forest is the first of the three machine learning algorithms I choose. This
algorithm can be used for both classification and regression. Random forest uses a
set of decision trees and picks the mode result of them to classify a data point. This
algorithm should cover for overfitting often done on singular decision trees.

Decision Tree is the second of the three machine learning algorithms I choose. This
algorithm like the first can be used for both classification and regression. In essence
this algorithm is what random forest is built on. I wanted to see how much benefit I
would get from using the more complicated random forest model over this one.

K Nearest Neighbours is the final model I choose and like the others can be both
used for classification and regression. This model works by mapping a data point to
the average values of the closest other data points.
