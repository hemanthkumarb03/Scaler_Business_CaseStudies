#Credits: https://medium.com/@enozeren/building-a-decision-tree-from-scratch-324b9a5ed836
import numpy as np 

class TreeNode():
    """
        To Create a tree we need a basic node structure where it has left and right nodes.
    """
    def __init__(self, data, feature_idx, feature_value, prediction_probs, information_gain) -> None:
        self.data = data
        self.feature_idx = feature_idx
        self.feature_value = feature_value
        self.prediction_probs = prediction_probs
        self.information_gain = information_gain
        self.left = None 
        self.right  = None 
        self.feature_importance = self.data.shape[0] * self.information_gain 

    def node_def(self) -> None:
        if (self.left or self.right):
            return f"Node | Information Gain = {self.information_gain} | Split if X[{self.feature_idx}] < {self.feature_val} Then left o/w right"
        else:
            unique_values, value_counts = np.unique(self.data[:,-1],return_counts=True)
            output = ", ".join([f"{value}->{count}" for value,count in zip(unique_values,value_counts)])
            return f"LEAF | Label Counts = {output} | Pred Probs = {self.prediction_probs}"
        

