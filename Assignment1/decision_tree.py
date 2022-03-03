from common import *
import matplotlib.pyplot as plt

class Node:
    def __init__(self, data_indices, parent):
        self.data_indices = data_indices
        self.left = None
        self.right = None
        self.split_feature = None
        self.split_value = None
        self.depth = 0
        self.data = None
        self.probability_of_each_class = None
        if parent:
            self.depth = parent.depth + 1


class DecisionTree:
    def __init__(self, cost_function='misclassification', max_depth=3, min_leaf_instances=1):
        self.max_depth = max_depth
        self.root = None
        self.const_fn_name = cost_function
        if cost_function == 'misclassification':
            self.cost_function = self.misclassification_cost
        elif cost_function == 'entropy':
            self.cost_function = self.entropy_cost
        elif cost_function == 'gini':
            self.cost_function = self.gini_index_cost
        self.num_classes = None
        self.min_leaf_instances = min_leaf_instances
        self.data = None
        self.labels = None

    def __str__(self):
        return f'Decision tree with cost fn = {self.const_fn_name}, ' \
               f'max depth = {self.max_depth}, min leaf instances = {self.min_leaf_instances} '

    @staticmethod
    def decision_boundary(data, y_label, model):
        x = data.loc[:, data.columns != y_label].to_numpy()
        y = data.loc[:, data.columns == y_label].to_numpy().ravel()
        x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
        y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
        model.fit(x, y)
        y_pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
        y_pred = np.argmax(y_pred, 1)
        y_pred = y_pred.reshape(xx.shape)
        cs = plt.contourf(xx, yy, y_pred, cmap=plt.cm.Pastel1)
        plt.suptitle("Decision surface of decision trees for Hepatitis Dataset")
        for i, color in zip(range(1, 3), "rb"):
                idx = np.where(y == i)
                plt.scatter(
                    x[idx, 0],
                    x[idx, 1],
                    c=color,
                    cmap=plt.cm.RdYlBu,
                    s=10,
                    alpha=0.5
                )
    @staticmethod
    def misclassification_cost(data_points):
        return 1 - np.max(np.bincount(data_points) / np.sum(np.bincount(data_points)))

    @staticmethod
    def entropy_cost(data_points):
        probability_of_each_class = np.bincount(data_points) / len(data_points)
        probability_of_each_class = probability_of_each_class[probability_of_each_class > 0]
        return -np.sum(probability_of_each_class * np.log(probability_of_each_class))

    @staticmethod
    def gini_index_cost(data_points):
        probability_of_each_class = np.bincount(data_points) / len(data_points)
        return 1 - np.sum(np.square(probability_of_each_class))

    def split(self, node):
        best_cost = np.inf
        best_feature = None
        best_test_value = None

        # Sort the data points in ascending order
        data = self.data[node.data_indices]
        sorted_data = np.sort(data, axis=0)

        number_of_instances = data.shape[0]
        number_of_features = data.shape[1]

        # Calculating average values of adjacent data points
        candidates = (sorted_data[1:] + sorted_data[:-1]) / float(2)

        # Calculate the best feature and a value from the feature
        # to use for the branching
        for feature in range(number_of_features):
            feature_data_points = data[:, feature]
            # Test for average values between each
            # data points in the feature
            for value in candidates[:, feature]:
                left_indices = node.data_indices[feature_data_points <= value]
                right_indices = node.data_indices[feature_data_points > value]

                # If a value causes either left or right sub array to be 0
                # then the best cost is inf
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                # Calculate a weighted cost, using the given cost function
                left_cost = self.cost_function(self.labels[left_indices])
                right_cost = self.cost_function(self.labels[right_indices])
                cost = (left_indices.shape[0] * left_cost + right_indices.shape[0] * right_cost) / number_of_instances

                # Update the cost if there is a better cost
                if cost < best_cost:
                    best_cost = cost
                    best_feature = feature
                    best_test_value = value

        return best_cost, best_feature, best_test_value

    def _fit_tree(self, node):
        # Calculate the probability of reaching this node
        probability_of_each_class = np.bincount(self.labels[node.data_indices], minlength=self.num_classes)
        node.probability_of_each_class = probability_of_each_class / np.sum(probability_of_each_class)

        # Stopping criteria
        # (i) maximum depth
        # (ii) minimum number of elements in the leaf nodes
        # (iii) if entropy, calculate the improvement in the entropy
        if node.depth == self.max_depth \
                or len(node.data_indices) <= self.min_leaf_instances:
            return

        # Get the optimal split for the data points associated with the node
        cost, node.split_feature, node.split_value = self.split(node)

        # If the optimal cost is inf, then there's nothing more to do
        if np.isinf(cost):
            return

        # Get the data points towards the left and right of the split respectively
        test = self.data[node.data_indices, node.split_feature] <= node.split_value
        node.left = Node(node.data_indices[test], node)
        node.right = Node(node.data_indices[np.logical_not(test)], node)

        # Run the _fit_tree method recursively for the individual splits
        self._fit_tree(node.left)
        self._fit_tree(node.right)

    def fit(self, data, labels):
        self.data = data
        self.labels = labels
        self.num_classes = np.max(self.labels) + 1
        self.root = Node(np.arange(data.shape[0]), None)
        self._fit_tree(self.root)

    def predict(self, test_data):
        # Initialise the probabilities with 0
        probability_of_each_class = np.zeros((test_data.shape[0], self.num_classes))
        # For every data point in the test data
        # follow the tree branching until you reach the leaf
        # finally calculate the probability
        for n, x in enumerate(test_data):
            node = self.root
            while node.left:
                if x[node.split_feature] <= node.split_value:
                    node = node.left
                else:
                    node = node.right
            probability_of_each_class[n, :] = node.probability_of_each_class
        return np.argmax(probability_of_each_class, 1)