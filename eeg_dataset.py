import torch


class EEGDataset(torch.utils.data.Dataset):
    """
    Create EEG dataset
    """
    def __init__(self, signal, stage):
        """
        Initializing dataset
        :param signal: Input signal
        :param stage: Stages
        """
        self.stages = stage
        self.eeg = signal

    def __len__(self):
        """
        Return number of elements in dataset
        :return: Number of elements in dataset
        """
        return len(self.stages)

    def __getitem__(self, idx):
        """
        Return item from dataset by index
        :param idx: index of element
        :return: Element by index - tuple: eeg signal and its stage
        """
        eeg = self.eeg[idx]
        stage = self.stages[idx]
        return eeg, stage
