# Necessary imports
import pandas as pd
import numpy as np
import pickle
from IPython.display import display
from tqdm import notebook

class Node():
    def __init__(self, value=None, info_gain=None, threshold=None, left=None, right=None, feature=None):
        # Leaf nodes
        self.value = value
        
        # Decision nodes
        self.info_gain = info_gain
        self.threshold = threshold
        self.left = left
        self.right = right
        self.feature = feature

class DecisionTree():
    def __init__(self, min_samples_split=2, max_depth=5):
        self.min_samples_split = min_samples_split # If number of samples is less than this then don't split
        self.max_depth = max_depth # Maximum depth of the tree
        self.root = None
    
    def build_tree(self, dataset, curr_depth=0, method='gini'):
        """ Function to build a binary tree """        
        # Retrieve features and labels
        X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1] 
        
        # Set some base conditions for when a dataset is uniform
        if len(y.value_counts()) == 1:
            return Node(self.calculate_leaf_value(y))
        
        num_samples, num_features = np.shape(X)
        
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.best_split(dataset, num_samples, num_features, method=method)
            
            if best_split is None:
                pickle.dump(dataset, open("none_best_split.pkl", "wb"))
                return Node(self.calculate_leaf_value(y))
            
            if 'info_gain' not in best_split.keys():
                display(dataset)
            if best_split['info_gain'] > 0:
                # Recursively build the left and right sub trees
                left_child = self.build_tree(best_split['dataset_left'], curr_depth + 1)
                right_child = self.build_tree(best_split['dataset_right'], curr_depth + 1)
            
                # Return a decision node
                return Node(info_gain=best_split['info_gain'], 
                            threshold=best_split['threshold'],
                            left=left_child, right=right_child,
                            feature=best_split['feature']) 
        
        # Leaf nodes if: info_gain = 0 or reached min_samples_split or reached max_depth
        return Node(self.calculate_leaf_value(y))
        
        
    def best_split(self, dataset, num_samples, num_features, method='gini'):
        """ Return a dictionary "best_split" containing the split datasets, feature, threshold, and the info_gain """
        best_split = {}
        max_info_gain = -float("inf")
        
        for feature in dataset.iloc[:, :-1]:
            possible_thresholds = dataset[feature]
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self.split(dataset, feature, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    
                    y = dataset.iloc[:, -1]
                    left_y = dataset_left.iloc[:, -1]
                    right_y = dataset_right.iloc[:, -1]
                    
                    curr_ig = self.calculate_info_gain(y, left_y, right_y, method)
                    if curr_ig >= max_info_gain:
                        best_split['dataset_right'] = dataset_right
                        best_split['dataset_left'] = dataset_left
                        best_split['threshold'] = threshold
                        best_split['feature'] = feature
                        best_split['info_gain'] = curr_ig
                        max_info_gain = curr_ig
                        
        # If no best split
        if best_split == {}:
            return None
        return best_split
                        
    def split(self, dataset, feature, threshold):
        """ Returns a left and right dataset after a split """
        dataset_left = dataset.loc[dataset[feature] <= threshold]
        dataset_right = dataset.loc[dataset[feature] > threshold]
        return dataset_left, dataset_right
        
    def calculate_leaf_value(self, labels):
        return labels.mode()[0]
        
    def calculate_info_gain(self, parent, left_child, right_child, method='gini'):
        weight_l = len(left_child) / len(parent)
        weight_r = len(right_child) / len(parent)
        if method == 'entropy':
            return entropy(parent) - weight_l * entropy(left_child) - weight_r * entropy(right_child)
        elif method == 'gini':
            gini_parent = gini(parent)
            gini_left = gini(left_child)
            gini_right = gini(right_child)
            return gini(parent) - weight_l * gini(left_child) - weight_r * gini(right_child)
        else:
            raise("Unrecognized cost calculation function")
        
    def shuffle(self, X_train, y_train):
        X_train['labels'] = y_train
        new = X_train.sample(frac=1)
        new_X, new_y = new.iloc[:, :-1], new.iloc[:, -1]
        return (new_X, new_y)
    
    def fit(self, X_train, y_train, cost_func='gini'):
        dataset = X_train.copy()
        dataset['labels'] = y_train
        self.root = self.build_tree(dataset, method=cost_func)
        
    
    def display_tree(self, root, depth=0):
        if root is None:
            return
        self.print_node(root, depth)
        self.display_tree(root.right, depth + 1)
        self.display_tree(root.left, depth + 1)
                
    def predict_singular(self, X, node=None):
        if node is None:
            node = self.root
        if node.value is None: # decision node:
            if X[node.feature] >= node.threshold:
                return self.predict_singular(X, node.right)
            else:
                return self.predict_singular(X, node.left)
        else:
            return node.value
    
    def predict(self, X):
        preds = [self.predict_singular(x) for ind, x in X.iterrows()]
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

    def evaluate(self, X_test, y_test):
        score = 0
        y_pred = self.predict(X_test)
        for x, y in zip(y_pred, list(y_test)):
            if x == y: score += 1
    
        test_size = len(y_test)
        print(f"Score: {score}/{test_size}")

        
        cm = self._get_confusion_matrix(y_pred, y_test)
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
        
    
    def print_node(self, node, depth):
        string = "--" * depth
        if node.value is None:
            print(string + f"Decision node: {node.feature} > {node.threshold}")
        else:
            print(string + f"Leaf node: {node.value}")

def gini(feature):
    if isinstance(feature, pd.Series):
        probs = feature.value_counts() / feature.shape[0]
        gini = 1 - np.sum(probs**2)
        return gini
    else:
        raise("Object must be a Pandas Series")

def entropy(feature):
    if isinstance(feature, pd.Series):
        probs = feature.value_counts() / feature.shape[0]
        return np.sum((-1) * probs * np.log2(probs + 1e-9))
    else:
        raise("Object must be a Pandas Series")