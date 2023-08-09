import numpy as np
import pandas as pd
import collections

from decision_tree import DecisionTree

class RandomForest:
    def __init__(self, num_trees=5, max_depth=5, min_sample_split=2):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.trees = []
    
    def fit(self, X_train, y_train, cost_func='gini'):
        self.trees = []
        for _ in range(self.num_trees):
            tree = DecisionTree(min_samples_split=self.min_sample_split, max_depth=self.max_depth)
            sample_X, sample_y = self._bootstrap_sampling(X_train, y_train)
            tree.fit(X_train=sample_X, y_train=sample_y, cost_func=cost_func)
            self.trees.append(tree)

    def _bootstrap_sampling(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return (X.iloc[indices], y.iloc[indices])

    def predict(self, X):
        preds = []
        for tree in self.trees:
            preds.append(tree.predict(X))

        tree_preds = np.swapaxes(preds, 0, 1)
        preds = []
        for pred in tree_preds:
            preds.append(pd.Series(pred).mode()[0])
        return preds
    
    def _get_confusion_matrix(self, preds, labels):
        # Labels: 0 means valid entry, 1 means false entry
        # Positive class: valid entry
        # Negative class: false entry
        # Return a matrix as such: [[TP, FP], [FN, TN]]
        tp, fp, fn, tn = 0, 0, 0, 0
        for pred, label in zip(list(preds), list(labels)):
            if pred == label and pred == 0:
                tp += 1
            elif pred == label and pred == 1:
                tn += 1
            elif pred != label and label == 0:
                fn += 1
            elif pred != label and label == 1:
                fp += 1
        return [[tp, fp], [fn, tn]]
    
    def evaluate(self, X, y):
        score = 0
        preds = self.predict(X)
        for pred, label in zip(preds, list(y)):
            if pred == label:
                score += 1

        test_size = len(y)
        print(f"Score: {score}/{test_size}")

        cm = self._get_confusion_matrix(preds, y)
        print("Confusion matrix: " + str(cm))
        
        # Calculate the different metrics
        tp = cm[0][0]
        tn = cm[1][1]
        fp = cm[0][1]
        fn = cm[1][0]
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = (tp) / (tp + fp)
        recall = (tp) / (tp + fn)
        f1 = (2 * tp) / (2 * tp + fp + fn)
        print("Accuracy:", accuracy) 
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 score:", f1)
