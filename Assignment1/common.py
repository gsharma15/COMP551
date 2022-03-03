import pandas as pd
import numpy as np
"""
Contains common code for both KNN and decision trees.
"""


def hep_load(file_path, model) -> pd.DataFrame:
    HEP_DEAD = 1  # 1 means dead
    HEP_ALIVE = 2  # 2 means alive


    # Based on info in hepatitis.names
    col_names = ['class', 'age', 'sex', 'steroid', 'antivirals', 'fatigue', 'malaise',
                 'anorexia', 'liver big', 'liver firm', 'spleen palpable', 'spiders', # wtf??
                  'ascites', 'varices', 'bilirubin', 'alk phosphate', 'sgot', 'albumin',
                 'protime', 'histology']

    df = pd.read_csv(file_path, names=col_names)

    # Sanity check some basic info as per hepatitis.names
    assert len(df.columns) == 20
    assert len(df) == 155
    assert len(df[df['class'] == HEP_DEAD]) == 32
    assert len(df[df['class'] == HEP_ALIVE]) == 123

    # This is important to convert everything to floats. Values that can't be converted to
    # numbers will be NaNs.
    df = df.apply(pd.to_numeric, errors='coerce')

    if model == 'knn':
        df = (df - df.min()) / (df.max() - df.min())
    return df


def hep_clean(df):
    return df.dropna()


def dr_load(file_path, model) -> pd.DataFrame:
    col_names = ['quality assesment', 'retinal abnormality',
                 'MAs @ confidence level 0.5', 'MAs @ confidence level 0.6',
                 'MAs @ confidence level 0.7', 'MAs @ confidence level 0.8',
                 'MAs @ confidence level 0.9', 'MAs @ confidence level 1.0',
                 'exudates 1', 'exudates 2', 'exudates 3', 'exudates 4',
                 'exudates 5', 'exudates 6', 'exudates 7', 'exudates 8',
                 'macula-optic disc distance', 'optic disc diameter',
                 'AM/FM', 'class']

    df = pd.read_csv(file_path, skiprows=24, names=col_names)

    # See https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set#
    # for a description of this dataset.
    assert len(col_names) == 20
    assert len(df) == 1151

    if model == 'knn':
        df = (df - df.min()) / (df.max() - df.min())

    return df


def dr_clean(df):
    # There's no missing data in the diabetes dataset
    return df.dropna()


def evaluate_acc(y_pred: np.ndarray, y_actual: np.ndarray) -> float:
    # Returns accuracy %.
    num_correct_labels = np.sum(y_pred == y_actual)
    return (num_correct_labels / len(y_pred)) * 100.0


def cross_validate(model, all_data: pd.DataFrame, L=3) -> float:
    '''
    Returns L-folds cross-validated accuracy %.
    '''
    total_size = len(all_data)
    # It's possible for L to not be an exact multiple of total_size, but
    # in such cases the number of dropped datapoints will be very small
    # compared to total data points.
    fold_size = total_size // L

    # Extract labels as a separate array
    labels = all_data['class']
    # Remove labels from data array
    all_data.drop('class', axis=1, inplace=True)
    total_accuracy = 0.0

    for l in range(L):
        # Create empty dataframes to hold training set
        training_data = pd.DataFrame(dtype='float64', columns=all_data.columns)
        training_labels = pd.Series(dtype='int64')

        validation_data = training_data = pd.DataFrame(dtype='float64', columns=all_data.columns)
        validation_labels = pd.Series(dtype='int64')

        current_fold = 0
        current_start_index = 0
        while current_fold < L:
            end_index = current_start_index + fold_size
            fold_data = all_data[current_start_index : end_index]
            fold_labels = labels[current_start_index : end_index]
            # print(f'Sliced from {current_start_index} to {end_index}')

            if current_fold == l:
                # Hold out the l'th fold data for validation
                validation_data = fold_data
                validation_labels = fold_labels
            else:
                # else add the (l-1) folds to training set
                training_data = training_data.append(fold_data)
                training_labels = training_labels.append(fold_labels)

            current_fold += 1
            current_start_index = end_index

        # At this point we have the correct train and validation sets for the L'th run
        model.fit(training_data.to_numpy(), training_labels.to_numpy())
        y_pred = model.predict(validation_data.to_numpy())

        accuracy = evaluate_acc(y_pred, validation_labels.to_numpy())
        print(f'Accuracy for {model} for cross validation run {l} with folds size {fold_size} is {round(accuracy,2)}%.')
        total_accuracy += accuracy

    return total_accuracy / L
