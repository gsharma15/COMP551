from matplotlib.lines import Line2D

from knn import KNN
from decision_tree import DecisionTree
from common import *
import itertools
import matplotlib.pyplot as plt


def run_experiment(dataset, model, **kwargs):
    if dataset == 'diabetes':
        all_data = dr_load('datasets/diabetes/messidor_features.arff', model)
        split_index = int(len(all_data) * 0.6)  # 60/40% train/test split
        # No need to clean this as there are no missing values

        # Used for decision boundary plotting at the end
        F1 = 'MAs @ confidence level 0.5'
        F2 = 'MAs @ confidence level 0.6'
        true_label = 'Has signs of DR'
        false_label = 'No signs of DR'

    elif dataset == 'hepatitis':
        all_data = hep_load('datasets/hepatitis/hepatitis.data', model)
        all_data.drop('protime', axis=1, inplace=True) # drop protime column because most values are NaNs
        all_data = hep_clean(all_data)
        split_index = int(len(all_data) * 0.8)      # 80/20% train/test split because this is a smaller dataset

        # Used for decision boundary plotting at the end
        F1 = 'malaise'
        F2 = 'anorexia'
        true_label = 'Lived'
        false_label = 'Died'

    else:
        raise Exception('Wrong dataset name!')

    if model == 'dt':
        # Cross validation with grid search over various hyperparameters
        L = 3  # number of cross validation folds

        if kwargs['grid_search']:
            hyperparameters = {
                'cost_fn': ['misclassification', 'entropy', 'gini'],
                'max_depth': [1, 2, 5, 10, 50],
                'min_leaf_instances': [1, 2, 5, 10]
            }

            # Do a grid search of all hyperparameters
            keys, values = zip(*hyperparameters.items())
            grid_search_params = [dict(zip(keys, v)) for v in itertools.product(*values)]

            best_model = None
            best_validation_accuracy = -1.0

            # For report plots
            all_validation_accs = []
            all_models = []
            best_index = -1

            for index, params in enumerate(grid_search_params):
                # Get all the training data (training+validation sets)
                training_data = all_data[:split_index].copy()

                # Remaining data is for test set
                testing_data = all_data[split_index:].drop('class', axis=1)
                testing_labels = all_data['class'][split_index:]

                tree = DecisionTree(cost_function=params['cost_fn'],
                                    max_depth=params['max_depth'],
                                    min_leaf_instances=params['min_leaf_instances'])

                cross_accuracy = cross_validate(tree, training_data, L=L)
                print(f'Average {L} folds cross-validation accuracy '
                  f'for {tree} is {round(cross_accuracy, 2)}%.')

                all_validation_accs.append(cross_accuracy)
                all_models.append(tree)

                if cross_accuracy > best_validation_accuracy:
                    best_validation_accuracy = cross_accuracy
                    best_model = tree
                    best_index = index

            # Run the  best model on test set
            test_pred = best_model.predict(testing_data.to_numpy())
            test_acc = evaluate_acc(test_pred, testing_labels.to_numpy())

            print(f'{best_model} was found to be the best model with validation accuracy = '
                  f'{round(best_validation_accuracy, 2)}% and testing accuracy = {round(test_acc, 2)}%.')

            for_report = pd.DataFrame({'cost_fn': list(map(lambda x: x.const_fn_name, all_models)),
                                       'max depth': list(map(lambda x: x.max_depth, all_models)),
                                       'min_leaf_instances': list(map(lambda x: x.min_leaf_instances, all_models)),
                                       'validation accuracy %': all_validation_accs,
                                       'isBest': [test_acc if x == best_index else 0 for x in
                                                  range(len(all_validation_accs))]})
            for_report.to_csv(f'dt_{dataset}_report.csv', index=False)

    if model == 'knn':
        """
        # Vanilla train/test KNN
        knn = KNN(dist_fn=kwargs['distance_fn'], K=kwargs['K'])
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)
        accuracy = evaluate_acc(y_pred, y_test)
        print(f'Accuracy of {knn} on {dataset} dataset is {round(accuracy, 2)}%.')
        """

        #  Cross validation with grid search over all hyperparameters

        L = 3  # number of cross validation folds

        if kwargs['grid_search']:
            hyperparameters = {
                'Ks': [i for i in range(1, 20)],    # different values of K's to check for
                'dist_fn': [KNN.EUCLIDEAN, KNN.MANHATTEN]
            }

            best_model = None
            best_validation_accuracy = -1.0

            # For report plots
            all_validation_accs = []
            all_models = []
            best_index = -1

            # Do a grid search of all hyperparameters
            keys, values = zip(*hyperparameters.items())
            grid_search_params = [dict(zip(keys, v)) for v in itertools.product(*values)]

            for index, params in enumerate(grid_search_params):

                # Get all the training data (training+validation sets)
                training_data = all_data[:split_index].copy()

                # Remaining data is for test set
                testing_data = all_data[split_index:].drop('class', axis=1)
                testing_labels = all_data['class'][split_index:]

                knn = KNN(dist_fn=params['dist_fn'], K=params['Ks'])
                # Run cross validation
                cross_accuracy = cross_validate(knn, training_data, L=L)
                print(f'Average {L} folds cross-validation accuracy '
                      f'for {knn} is {round(cross_accuracy, 2)}%.')

                all_validation_accs.append(cross_accuracy)
                all_models.append(knn)

                if cross_accuracy > best_validation_accuracy:
                    best_validation_accuracy = cross_accuracy
                    best_model = knn
                    best_index = index

            # Run on test set
            test_pred = best_model.predict(testing_data.to_numpy())
            test_acc = evaluate_acc(test_pred, testing_labels.to_numpy())

            print(f'{best_model} was found to be the best model with validation accuracy = '
                  f'{round(best_validation_accuracy, 2)}% and testing accuracy = {round(test_acc, 2)}%.')


            for_report = pd.DataFrame({'K': list(map(lambda x: x.K, all_models)),
                                    'distance_fn': list(map(lambda x: x.dist_fn_name, all_models)),
                                  'validation accuracy %': all_validation_accs,
                                  'isBest': [test_acc if x == best_index else 0 for x in range(len(all_validation_accs))]})
            for_report.to_csv(f'knn_{dataset}_report.csv', index=False)


            # Decision boundary plotting code
            RESOLUTION = 50
            '''
            Top Absolute Correlations (hepatitis)
            malaise  anorexia    0.599647
            fatigue  malaise     0.595142
            ascites  albumin     0.531032
            
            Top Absolute Correlations (diabetes)
            MAs @ confidence level 0.5  MAs @ confidence level 0.6    0.996177
            MAs @ confidence level 0.6  MAs @ confidence level 0.7    0.994221
            MAs @ confidence level 0.7  MAs @ confidence level 0.8    0.991821
            MAs @ confidence level 0.8  MAs @ confidence level 0.9    0.988294
            MAs @ confidence level 0.5  MAs @ confidence level 0.7    0.985730
            '''
            # Discretize the feature space. Note that only 2 most correlated features are discretized,
            # and other feature values are filled with their respective mean values.
            mean_feature_values = all_data.drop('class', axis=1).mean(axis=0).to_frame().transpose()

            # Make copies of data to show on a discretized grid. Equidistant samples will be used for
            # the two most correlated features, and other features will be filled with mean values.

            # Replace the two features with generated data

            feature1v = np.linspace(np.min(all_data[F1]), np.max(all_data[F1]), RESOLUTION)
            feature2v = np.linspace(np.min(all_data[F2]), np.max(all_data[F2]), RESOLUTION)

            x, y = np.meshgrid(feature1v, feature2v)
            feature1v = x.ravel()
            feature2v = y.ravel()

            discretize_data = pd.concat([mean_feature_values] * len(feature1v), ignore_index=True)
            discretize_data[F1] = feature1v
            discretize_data[F2] = feature2v

            discretized_predictions = best_model.predict(discretize_data.to_numpy())

            # Hack to make hepatits class values be (0,1) instead of (1,2)
            if dataset == 'hepatitis':
                discretized_predictions = discretized_predictions - 1

            true_mask = discretized_predictions.astype(bool)
            false_mask = ~discretized_predictions.astype(bool)

            plt.scatter(feature1v[true_mask], feature2v[true_mask], c='red', label=true_label)
            plt.scatter(feature1v[false_mask], feature2v[false_mask], c='blue', label=false_label)

            plt.ylabel(F1)
            plt.xlabel(F2)
            plt.legend()
            plt.show()



    ''' sklearn comparison        
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    knn = KNeighborsClassifier(n_neighbors = K)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print('KNN accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

    '''


print('--------------------------------------------------------------------------------- Hepatitis dataset KNN ---------------------------------------------------------------------------------')
run_experiment('hepatitis', model='knn', grid_search=True)
print('--------------------------------------------------------------------------------- Diabetes dataset KNN ---------------------------------------------------------------------------------')
run_experiment('diabetes', model='knn', grid_search=True)

print('--------------------------------------------------------------------------------- Hepatitis dataset decision trees ---------------------------------------------------------------------------------')
run_experiment('hepatitis', model='dt', grid_search=True)
print('--------------------------------------------------------------------------------- Diabetes dataset decision trees ---------------------------------------------------------------------------------')
run_experiment('diabetes',  model='dt', grid_search=True)
