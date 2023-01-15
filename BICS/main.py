import mne
import numpy as np
from sklearn.model_selection import train_test_split
import preporocessing
from eeg_dataset import EEGDataset
import torch
from torch.utils.data import DataLoader
from neural_network import AutoEncoder
import train_and_eval_model
from sklearn.cluster import DBSCAN
from sklearn import metrics
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def save_results(sleep_stages, indexes_test, labels, y_test, path='result.csv'):
    """
    Save results to .csv format
    :param sleep_stages: dataframe with sleep_stages
    :param indexes_test: indexes of test data
    :param labels: labels from clusterisator
    :param y_test: labels from doctor
    :param path: path to file
    :return: file with results
    """
    sleep_stages_result = sleep_stages.iloc[indexes_test]
    result = {
        'start_time': sleep_stages_result['onset'].values,
        'end_time': sleep_stages_result['onset'].values + sleep_stages_result['duration'].values,
        'type_from_nn': labels,
        'type_from_doctor': y_test,
        'homogeneity': metrics.homogeneity_score(y_test, labels),
    }
    result = pd.DataFrame(result)
    result.to_csv(path, sep='\t')


def check_algorithm(estimator, d2_train_dataset, y_train, d2_test_dataset):
    """
    Check K-Neighbors algorithm
    :param estimator: Estimator (algorithm)
    :param d2_train_dataset: Train dataset with signals
    :param y_train: Train labels
    :param d2_test_dataset: Test dataset with signals
    :return: predicted labels
    """
    estimator.fit(d2_train_dataset, np.asarray(y_train, dtype=object).astype(np.int64))
    predict = estimator.predict(d2_test_dataset)
    return predict


def create_train_test_dataset(X_train, X_test, y_train, y_test):
    """
    Create train and test datasets
    :param X_train: Array of train signal
    :param X_test: Array of test signal
    :param y_train: Labels for train signal
    :param y_test: Labels for test signal
    :return: Train and test datasets
    """
    train_ds = EEGDataset(
        torch.tensor(np.stack
                     (np.asarray(X_train, dtype=object)).astype(np.float64), dtype=torch.double),
        torch.tensor(np.asarray(y_train, dtype=object).astype(np.int64), dtype=torch.int64))
    test_ds = EEGDataset(
        torch.tensor(np.stack(np.asarray(X_test, dtype=object)).astype(np.float64), dtype=torch.double),
        torch.tensor(np.asarray(y_test, dtype=object).astype(np.int64), dtype=torch.int))
    return train_ds, test_ds


if __name__ == '__main__':
    '''
    you should download to main directory these two files:
    https://physionet.org/content/sleep-edfx/1.0.0/sleep-cassette/SC4001E0-PSG.edf
    https://physionet.org/content/sleep-edfx/1.0.0/sleep-cassette/SC4001EC-Hypnogram.edf
    '''
    patient = 'SC4001E0-PSG.edf'
    annotation = 'SC4001EC-Hypnogram.edf'
    patient_1 = 'SC4002E0-PSG.edf'
    annotation_1 = 'SC4002EC-Hypnogram.edf'
    raw = mne.io.read_raw_edf(patient, preload=True)
    annot = mne.read_annotations(annotation)
    raw_1 = mne.io.read_raw_edf(patient_1, preload=True)
    annot_1 = mne.read_annotations(annotation_1)
    sleep_stages = preporocessing.preprocessing(annot, 60)
    eeg_signals = preporocessing.table_with_signals(sleep_stages, raw)
    indices = np.arange(len(eeg_signals['stage']))  # array of indexes
    sleep_stages_1 = preporocessing.preprocessing(annot_1, 60)
    eeg_signals_1 = preporocessing.table_with_signals(sleep_stages_1, raw_1)
    indices_1 = np.arange(len(eeg_signals_1['stage']))  # array of indexes
    """
    Divide data by train and test
    """
    X_train, X_test, y_train, y_test, indexes_train, indexes_test = train_test_split(eeg_signals['signal'],
                                                                                     eeg_signals['stage'], indices,
                                                                                     test_size=0.2,
                                                                                     stratify=eeg_signals['stage'])
    X_train_1, X_test_1, y_train_1, y_test_1, indexes_train_1, indexes_test_1 = train_test_split(
        eeg_signals_1['signal'],
        eeg_signals_1['stage'], indices_1,
        test_size=0.2,
        stratify=eeg_signals_1['stage'])
    """
    Create train and test datasets
    """
    train_ds, test_ds = create_train_test_dataset(X_train, X_test, y_train, y_test)
    train_ds_1, test_ds_1 = create_train_test_dataset(X_train_1, X_test_1, y_train_1, y_test_1)

    """
    Create train and test dataloaders
    """
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=64)
    test_dataloader = DataLoader(test_ds, shuffle=False, batch_size=128)
    train_dataloader_1 = DataLoader(train_ds_1, shuffle=True, batch_size=64)
    test_dataloader_1 = DataLoader(test_ds_1, shuffle=False, batch_size=128)

    """
    Initialize and train model
    """
    model = AutoEncoder(128)
    epoch = 50
    model = train_and_eval_model.train(model.double(), train_dataloader, epoch)

    """
    Check for different patients
    """

    """
    Evaluate model
    """
    autoencoder = model.eval()
    run_res = train_and_eval_model.run_eval(autoencoder, train_dataloader_1)

    """
    Clusterize labels
    """
    db = DBSCAN(eps=0.25, min_samples=2).fit(run_res['latent'])
    labels = db.labels_
    save_results(sleep_stages=sleep_stages_1, indexes_test=indexes_train_1,
                 labels=labels, y_test=y_train_1, path='result_different.csv')

    """
     Check other methods for different patients
     """
    n_samples, nx, ny = np.stack(np.asarray(X_train, dtype=object)).shape
    d2_train_dataset = np.stack(np.asarray(X_train, dtype=object)).reshape((n_samples, nx * ny))
    n_samples_test, nx_test, ny_test = np.stack(np.asarray(X_train_1, dtype=object)).shape
    d2_test_dataset = np.stack(np.asarray(X_train_1, dtype=object)).reshape((n_samples_test, nx_test * ny_test))

    """
    KNeighbors
    """
    estimator = KNeighborsClassifier()
    predict = check_algorithm(estimator, d2_train_dataset, y_train, d2_test_dataset)
    save_results(sleep_stages=sleep_stages_1, indexes_test=indexes_train_1,
                 labels=predict, y_test=y_train_1, path='result_k_neighbors_different.csv')

    """
    Random Forest Classifier
    """

    estimator = RandomForestClassifier(max_depth=100, random_state=0)
    predict = check_algorithm(estimator, d2_train_dataset, y_train, d2_test_dataset)
    save_results(sleep_stages=sleep_stages_1, indexes_test=indexes_train_1,
                 labels=predict, y_test=y_train_1, path='result_random_forest_different.csv')

    """
    Check for one patient
    """
    """
    Evaluate model
    """
    run_res = train_and_eval_model.run_eval(autoencoder, test_dataloader)

    """
    Clusterize labels
    """
    db = DBSCAN(eps=0.25, min_samples=2).fit(run_res['latent'])
    labels = db.labels_
    save_results(sleep_stages=sleep_stages, indexes_test=indexes_test,
                 labels=labels, y_test=y_test, path='result_same.csv')


    """
    Check other methods for the same patient
    """
    n_samples_test, nx_test, ny_test = np.stack(np.asarray(X_test, dtype=object)).shape
    d2_test_dataset = np.stack(np.asarray(X_test, dtype=object)).reshape((n_samples_test, nx_test * ny_test))

    """
    KNeighbors
    """
    estimator = KNeighborsClassifier()
    predict = check_algorithm(estimator, d2_train_dataset, y_train, d2_test_dataset)
    save_results(sleep_stages=sleep_stages, indexes_test=indexes_test,
                 labels=predict, y_test=y_test, path='result_k_neighbors_same.csv')

    """
    Random Forest Classifier
    """

    estimator = RandomForestClassifier(max_depth=100, random_state=0)
    predict = check_algorithm(estimator, d2_train_dataset, y_train, d2_test_dataset)
    save_results(sleep_stages=sleep_stages, indexes_test=indexes_test,
                 labels=predict, y_test=y_test, path='result_random_forest_same.csv')