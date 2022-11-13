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

if __name__ == '__main__':
    '''
    you should download to main directory these two files:
    https://physionet.org/content/sleep-edfx/1.0.0/sleep-cassette/SC4001E0-PSG.edf
    https://physionet.org/content/sleep-edfx/1.0.0/sleep-cassette/SC4001EC-Hypnogram.edf
    '''
    patient = 'SC4001E0-PSG.edf'
    annotation = 'SC4001EC-Hypnogram.edf'
    raw = mne.io.read_raw_edf(patient, preload=True)
    annot = mne.read_annotations(annotation)
    sleep_stages = preporocessing.preprocessing(annot, 60)
    eeg_signals = preporocessing.table_with_signals(sleep_stages, raw)
    indices = np.arange(len(eeg_signals['stage']))  # array of indexes
    """
    Divide data by train and test
    """
    X_train, X_test, y_train, y_test, indexes_train, indexes_test = train_test_split(eeg_signals['signal'],
                                                                                     eeg_signals['stage'], indices,
                                                                                     test_size=0.2,
                                                                                     stratify=eeg_signals['stage'])
    """
    Create train and test datasets
    """
    train_ds = EEGDataset(torch.tensor(np.stack(np.asarray(X_train, dtype=object)).astype(np.float64), dtype=torch.double),
                          torch.tensor(np.asarray(y_train, dtype=object).astype(np.int64), dtype=torch.int64))
    test_ds = EEGDataset(torch.tensor(np.stack(np.asarray(X_test, dtype=object)).astype(np.float64), dtype=torch.double),
                         torch.tensor(np.asarray(y_test, dtype=object).astype(np.int64), dtype=torch.int))

    """
    Create train and test dataloaders
    """
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=64)
    test_dataloader = DataLoader(test_ds, shuffle=False, batch_size=128)

    """
    Initialize autoencoder and train model
    """
    model = AutoEncoder(128)
    epoch = 20
    model = train_and_eval_model.train(model.double(), train_dataloader, epoch)

    """
    Evaluate model
    """
    autoencoder = model.eval()
    run_res = train_and_eval_model.run_eval(autoencoder, test_dataloader)

    """
    Clusterize labels
    """
    db = DBSCAN(eps=0.15, min_samples=4).fit(run_res['latent'])
    labels = db.labels_

    """
    Save results
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
    result.to_csv('result.csv', sep='\t')
