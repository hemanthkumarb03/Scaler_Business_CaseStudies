#Credits: https://medium.com/@enozeren/building-a-decision-tree-from-scratch-324b9a5ed836
import numpy as np
from collections import Counter
from treenode import TreeNode

class DecisionTree():
    """
    Decision Tree classifier
    """
    def __init__(self,max_depth=4,min_samples_leaf=4,min_information_gain=0.0,
                 number_of_features_splitting=None) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf  = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.number_of_features_splitting = number_of_features_splitting
        
    def _entropy(self,class_probs: list) -> float:
        return sum([-p*np.log2(p) for p in class_probs if p>0])
    
    def _class_probs(self, labels: list) -> float:
        total_count = len(labels)
        return [label_count / total_count for label_count in Counter(labels).values()]
    
    def _data_entropy(self, lables: list) -> float:
        return self._entropy(self._class_probs(lables))
    
    def _partition_entropy(self, subsets: list) -> float:
        #Finding weighted entropy at child nodes.
        total_count = sum([len(subset) for subset in subsets])
        return sum([self._data_entropy(subset) * (len(subset)/total_count) for subset in subsets])
    
    def _split(self, data: np.array, feature_idx: int, feature_val: float) -> tuple :
        mask_below_threshold = data[:,feature_idx] < feature_val 
        group1 = data[mask_below_threshold]
        group2 = data[~mask_below_threshold]
        return group1, group2 
    
    def _select_features_to_use(self,data: np.array) -> list:
        """
        Randomly selects the feature to use while splitting wrt number_of_features_splitting hyperparameter
        """
        feature_idx = list(range(data.shape[1]-1))

        if self.number_of_features_splitting == "sqrt":
            feature_idx_to_use  = np.rnadom.choice(feature_idx,size=int(np.sqrt(len(feature_idx))))
        elif self.number_of_features_splitting == "log" :
            feature_idx_to_use = np.random.choice(feature_idx, size=int(np.log2(len(feature_idx))))
        else:
            feature_idx_to_use = feature_idx 
        
        return feature_idx_to_use
    
    def _find_the_best_split(self,data: np.array) -> tuple:
        """
            Finds the best split with the lowest entropy given data
            Returns 2 splitted groups and split information
        """
        min_part_entropy = 1e9
        feature_idx_to_use = self._select_features_to_use(data)

        for idx in feature_idx_to_use:
            feature_vals = np.percentile(data[:,idx],q=np.arange(25,100,25))
            for feature_val in feature_vals:
                g1,g2 = self._split(data,idx,feature_val)
                part_entropy = self._partition_entropy([g1[:,-1],g2[:,-1]])
                if part_entropy < min_part_entropy:
                    min_part_entropy = part_entropy
                    min_entropy_feature_idx = idx 
                    min_entropy_feature_val = feature_val 
                    g1_min,g2_min = g1,g2 
        return g1_min,g2_min, min_entropy_feature_idx,min_entropy_feature_val, min_part_entropy
    
    def _find_label_probs(self,data: np.array) -> np.array:
        labels_as_integers = data[:,-1].astype(int) 
        total_labels = len(labels_as_integers)
        label_probs = np.zeros(len(self.lables_in_train),dtype=float)

        for i ,label in enumerate(self.lables_in_train):
            label_index = np.where(labels_as_integers == i)[0]
            if len(label_index) > 0:
                label_probs[i] = len(label_index) / total_labels
        return label_probs
    
    def _create_tree(self,data: np.array,current_depth: int) -> TreeNode:
        """Recursive depth first tree creation algorithm"""
        #check if max depth has been reached
        if current_depth > self.max_depth:
            return None 
        
        #find the best split
        split_1_data, split_2_data, split_feature_idx, split_feature_val, split_entropy = self._find_best_split(data)

        #find the label probs for the node
        label_probs = self._find_label_probs(data)

        #calculate information gain
        node_entropy = self._entropy(label_probs)
        information_gain = node_entropy - split_entropy 

        #create node
        node = TreeNode(data,split_feature_idx,split_feature_val,label_probs,information_gain)

        #stoppin criteria: check if the min_samples_leaf has been satisfied
        if self.min_samples_leaf > split_1_data.shape[0] or self.min_samples_leaf > split_2_data.shape[0]:
            return node 
        elif information_gain < self.min_information_gain:
            return node 
        
        current_depth +=1
        node.left = self._create_tree(split_1_data,current_depth)
        node.right = self._create_tree(split_2_data,current_depth)

        return node 
    
    def _predict_one_sample(self,X:np.array) -> np.array:
        #prediction for one dimension
        node = self.tree 

        #finds the leaf which x belongs
        while node:
            pred_probs = node.prediction_probs 
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right
        return pred_probs 
    
    def train(self,X_train:np.array, Y_train: np.array) -> None:
        "Model training"

        self.lables_in_train = np.unique(Y_train)
        train_data = np.concatenate((X_train,np.reshape(Y_train,(-1,1))),axis=1)

        #start creating tree:
        self.tree = self._create_tree(data=train_data,current_depth=0)

        #calculate feature importance
        self.feature_importances = dict.fromkeys(range(X_train.shape[1]),0)
        self._calculate_feature_importance(self.tree)

        #normalize feature importance values
        self.feature_importances = {k:v /total for total in (sum(self.feature_importances.values()),) for k,v in self.feature_importances.items()}

    def predict_probs(self,X_set: np.array) -> np.array:
        return np.apply_along_axis(self._predict_one_sample,1,X_set)
    
    def predict(self, X_set: np.array) -> np.array:
        pred_probs = self.predict_probs(X_set)
        return np.argmax(pred_probs, axis=1)
    
    def _print_recursive(self, node: TreeNode, level=0) -> None:
        if node != None:
            self._print_recursive(node.left, level + 1)
            print('    ' * 4 * level + '-> ' + node.node_def())
            self._print_recursive(node.right, level + 1)

    def print_tree(self) -> None:
        self._print_recursive(node=self.tree)

    def _calculate_feature_importance(self, node):
        """Calculates the feature importance by visiting each node in the tree recursively"""
        if node != None:
            self.feature_importances[node.feature_idx] += node.feature_importance
            self._calculate_feature_importance(node.left)
            self._calculate_feature_importance(node.right)  
