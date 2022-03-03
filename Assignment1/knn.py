from common import *

class KNN:
    EUCLIDEAN = lambda x1, x2: np.sqrt(np.sum((x1 - x2) ** 2, axis=-1))
    MANHATTEN = lambda x1, x2: np.sum(np.abs(x1 - x2), axis=-1)

    def __init__(self, K, dist_fn):
        self.K = K
        self.dist_fn = dist_fn

        if dist_fn == KNN.EUCLIDEAN:    self.dist_fn_name = 'euclidean'
        elif dist_fn == KNN.MANHATTEN:  self.dist_fn_name = 'manhatten'

    def __str__(self):
        return f'KNN model with K = {self.K} using {self.dist_fn_name} distance function'

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Just store the data and class labels
        self.X_train = X
        self.y_train = y

    def predict(self, X_test: np.ndarray) -> np.ndarray:

        X_test = X_test
        n_test = len(X_test)
        '''
        We want to find distance of every point in X_test of shape (n_test,n_features)
        with every point in X_train of shape (n_train, n_features). To do this in one 
        numpy operation, we pad X_test with a dummy axis so that its shape is 
        (n_train, 1, n_features).
        See https://stackoverflow.com/a/37903795/3613100 for a good explanation.
        distances will be of shape (n_test, n_train) and will contain distance of each point
        in X_test (axis 0) with each point in X_train (axis 1). 
        '''
        distances = self.dist_fn(X_test[:, np.newaxis, :], self.X_train)

        '''
        Initialize a matrix to store KNN's indices for each test datapoint. This will have shape 
        (n_test, K). The i'th row will store indices of KNN's to the i'th test point. The 
        indices refer to indices in self.X_train.
        '''
        knns_indices = np.zeros((n_test, self.K), dtype=int)

        # i'th entry in y_pred will be the predicted class label of i'th test datapoint.
        y_pred = np.zeros(n_test, dtype=int)

        for i in range(n_test):
            knns_indices[i, :] = np.argsort(distances[i])[:self.K]
            knn_classes = self.y_train[knns_indices[i]]
            vals, counts = np.unique(knn_classes, return_counts=True)
            y_pred[i] = vals[np.argmax(counts)]

        return y_pred



