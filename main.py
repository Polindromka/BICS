import mne
import pandas as pd
import preporocessing
# Press the green button in the gutter to run the script.

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


