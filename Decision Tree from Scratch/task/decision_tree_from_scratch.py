import pandas as pd
from collections import Counter
import random
from sklearn.metrics import confusion_matrix

# train_data_file = "test/data_stage9_train.csv"
# test_data_file = "test/data_stage9_test.csv"
target_feature = "Survived"


class Node:
    def __init__(self):
        # class initialization
        self.left = None
        self.right = None
        self.term = False
        self.label = None
        self.feature = None
        self.value = None

    def __str__(self):
        if self.term:
            text = f"leaf node: label = {self.label}"
        else:
            text = f"feature = {self.feature}, value = {self.value}"
        return text

    def set_split(self, feature, value):
        # this function saves the node splitting feature and its value
        self.feature = feature
        self.value = value

    def set_term(self, label):
        # if the node is a leaf, this function saves its label
        self.term = True
        self.label = label


class DecisionTree:
    min_samples_for_leaf: int = 74  # 74

    def __init__(self, features_samples: dict, target_samples: list, float_types: dict):
        self.features_samples = features_samples
        self.target_samples = target_samples
        self.float_types = float_types
        self.decision_tree: Node | None = None

    def __str__(self):
        """ print tree structure """
        print("root node: ", self.decision_tree)
        self._print_tree(self.decision_tree)
        return ""

    def _print_tree(self, node: Node, level: int = 1) -> None:
        """ Recursive print tree """
        if (node.left is not None) and (node.right is not None):
            print("\t" * level + "true:  ", node.left)
            self._print_tree(node.left, level + 1)
            print("\t" * level + "false: ", node.right)
            self._print_tree(node.right, level + 1)

    def fit(self) -> None:
        self.decision_tree = Node()
        if self._is_leaf():
            self.decision_tree.set_term(self._get_target_label())
        else:
            self._add_nodes(self.decision_tree)

    def predict(self, test_data: list[dict]) -> list:
        """ returns list of predicted values for test_data """
        predictions = []
        for i, data in enumerate(test_data):
            # print(f"Prediction for sample # {i}")
            predictions.append(self._get_prediction(data, self.decision_tree))
        return predictions

    def _get_prediction(self, data: dict, node: Node):
        """ use decision tree to get predicted value for data """
        if node.term:  # node is leaf => return label
            # print(f"\tPredicted label: {node.label}")
            return node.label
        # print(f"\tConsidering decision rule on feature {node.feature} with value {node.value}")
        if self.float_types[node.feature]:
            if data[node.feature] <= node.value:
                # left node
                return self._get_prediction(data, node.left)
            else:
                # right node
                return self._get_prediction(data, node.right)
        else:
            if data[node.feature] == node.value:
                # left node
                return self._get_prediction(data, node.left)
            else:
                # right node
                return self._get_prediction(data, node.right)

    def _add_nodes(self, parent_node: Node) -> None:
        # split observations and add child notes
        if self._is_leaf():  # base case
            parent_node.set_term(self._get_target_label())
        else:  # recursive split nodes
            split_data = self._get_split_data()

            # !!!
            r_index = []
            for i, x in enumerate(self.target_samples):
                if i not in split_data[3]:
                    r_index.append(i)
            # print(split_data[0], split_data[1], round(split_data[2],3), split_data[3], r_index)
            # sys.exit()
            features_samples_l, target_samples_l, features_samples_r, target_samples_r = self._samples_split(
                split_data[3])
            parent_node.set_split(split_data[1], split_data[2])  # [1] = feature, [2] = feature level
            # print(f"Made split: {split_data[1]} is {split_data[2]}")
            # add left node
            parent_node.left = Node()
            self.features_samples = features_samples_l
            self.target_samples = target_samples_l
            self._add_nodes(parent_node.left)
            # add right node
            parent_node.right = Node()
            self.features_samples = features_samples_r
            self.target_samples = target_samples_r
            self._add_nodes(parent_node.right)

    def _get_split_data(self) -> list:
        """ get weighted Gini, feature, feature level, and left node indices for minimum weighted Gini index"""
        weighted_gini_indices: list = []
        node_data: list = []
        for feature in self.features_samples:
            feature_samples = self.features_samples[feature]
            counts = Counter(feature_samples)
            for level in counts:
                # split by level and compute weighted gini of target feature
                if self.float_types[feature]:
                    nodes_left, target_left, target_right = self._split_target_by_feature_level_float(feature_samples,
                                                                                                      level,
                                                                                                      self.target_samples)
                else:
                    nodes_left, target_left, target_right = self._split_target_by_feature_level(feature_samples, level,
                                                                                                self.target_samples)
                weighted_gini_indices.append(self._weighted_gini(target_left, target_right))
                node_data.append([self._weighted_gini(target_left, target_right), feature, level, nodes_left])

        index_min = weighted_gini_indices.index(min(weighted_gini_indices))
        return node_data[index_min]

    @staticmethod
    def _split_target_by_feature_level(feature_samples, feature_level, target_sample):
        nodes_left = []
        target_left = []
        target_right = []
        for i, f in enumerate(feature_samples):
            if f == feature_level:
                nodes_left.append(i)
                target_left.append(target_sample[i])
            else:
                target_right.append(target_sample[i])
        return nodes_left, target_left, target_right

    @staticmethod
    def _split_target_by_feature_level_float(feature_samples, feature_level, target_sample):
        nodes_left = []
        target_left = []
        target_right = []
        for i, f in enumerate(feature_samples):
            if f <= feature_level:
                nodes_left.append(i)
                target_left.append(target_sample[i])
            else:
                target_right.append(target_sample[i])
        return nodes_left, target_left, target_right

    def _samples_split(self, left_node_indices: list) -> tuple[dict, list, dict, list]:
        # split samples by node indices
        split_targets_l = []
        split_targets_r = []
        split_features_l = {key: [] for key in self.features_samples.keys()}
        split_features_r = {key: [] for key in self.features_samples.keys()}
        for i in range(len(self.target_samples)):
            if i in left_node_indices:
                split_targets_l.append(self.target_samples[i])
                for feature in self.features_samples:
                    split_features_l[feature].append(self.features_samples[feature][i])
            else:
                split_targets_r.append(self.target_samples[i])
                for feature in self.features_samples:
                    split_features_r[feature].append(self.features_samples[feature][i])
        return split_features_l, split_targets_l, split_features_r, split_targets_r

    def _get_target_label(self):
        # get most likely value of target_samples
        # in case of more than one different value with equal likelihoods get random value of most likely values
        counts = dict(Counter(self.target_samples).most_common())
        most_likely_value = list(counts)[0]
        most_likely_value_count = counts[most_likely_value]
        # check if this value is unique
        unique = 0
        for val in counts:
            if counts[val] < most_likely_value_count:
                break
            unique += 1
        if unique == 1:  # unique most_likely_value
            return most_likely_value
        else:
            # get all most likely values and choose random
            most_likely_values_list = list(counts)[:unique]
            return random.choice(most_likely_values_list)

    def _is_leaf(self) -> bool:
        # check leaf criteria
        # 1. The amount of data in this node is less or equal to the specified minimum. In this stage, the minimum is 1.
        if len(self.target_samples) <= self.min_samples_for_leaf:
            return True
        # 2. The Gini Impurity is 0.
        # => we have unique target feature
        if len(set(self.target_samples)) == 1:
            return True
        # 3. All objects (one dataset row, also known as an observation) have the same values for all features.
        different_features = []
        for feature in self.features_samples:
            different_features.append((len(set(self.features_samples[feature]))))
        if len(set(different_features)) == 1 and different_features[0] == 1:
            return True
        return False

    def _weighted_gini(self, sample1: list, sample2: list) -> float:
        gini1 = self._gini(sample1)
        gini2 = self._gini(sample2)
        res = (len(sample1) * gini1 + len(sample2) * gini2) / (len(sample1) + len(sample2))
        return round(res, 5)

    @staticmethod
    def _gini(sample: list) -> float:
        # determine how many different levels we have in our sample and count them
        sample_len = len(sample)
        counts = Counter(sample)
        gini_index = 1
        for level in counts:
            gini_index -= (counts[level] / sample_len) ** 2
        return gini_index


def main() -> None:
    files = input().split(" ")
    train_data_file = files[0]
    test_data_file = files[1]

    target_samples, features_samples, float_types = read_train_data(train_data_file, target_feature)
    decision_tree_model = DecisionTree(features_samples, target_samples, float_types)
    decision_tree_model.fit()

    test_data = pd.read_csv(test_data_file, index_col=0)
    test_data_records = test_data.to_dict(orient="records")
    predictions = decision_tree_model.predict(test_data_records)
    test_data_features = test_data.to_dict(orient="list")[target_feature]
    # now compare predictions with true values from test_data_features
    conf_mat = confusion_matrix(test_data_features, predictions, normalize="true")
    print(round(conf_mat[1][1], 3), round(conf_mat[0][0], 3))
    # print(decision_tree_model)


def read_train_data(train_data_file: str, target_feature_: str) -> tuple[list, dict, dict]:
    # split train data into features and target
    # check features for numeric values
    train_data = pd.read_csv(train_data_file, index_col=0)
    data_types = train_data.dtypes.to_dict()
    data_types = {col: pd.api.types.is_float_dtype(data_types[col]) for col in data_types}
    train_data = train_data.to_dict(orient="list")
    target_samples = train_data[target_feature_]
    train_data.pop(target_feature_)
    features_samples = train_data
    return target_samples, features_samples, data_types


if __name__ == "__main__":
    main()
