import torch
import torch.nn.functional as F
from torch import optim
import numpy as np


def train(model, train_dataloader, epoch=15):
    """
    Train our model
    :param model: Model of neural network (autoencoder)
    :param train_dataloader: Dataloader with train data
    :param epoch: number of epochs to train
    :return: trained model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    model.to(device)
    model.train()
    for epoch in range(epoch):
        for img_batch, labels in train_dataloader:
            optimizer.zero_grad()
            output, latent = model(img_batch.to(device))
            loss = F.mse_loss(output.to(device), torch.sigmoid(img_batch.to(device)))
            loss.backward()
            optimizer.step()
        print("Epoch {} Loss {:.4f}".format(epoch, loss.item()))
    return model


def run_eval(autoencoder, loader):
    """
    Evaluate model
    :param autoencoder: trained model (autoencoder)
    :param loader: dataloader (test data)
    :return: Result (dict: real signal, embeddings (main features), reconstruct signal, names of stages)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    real = []
    reconstruct_signal = []
    latent = []
    labels = []
    with torch.no_grad():
        for data, lab in loader:
            data = data.to(device)
            lab = lab.to(device)
            labels.append(lab.numpy())
            real.append(data.numpy())
            rec, emb = autoencoder(data.to(device))
            latent.append(emb.cpu().numpy())
            reconstruct_signal.append(rec.cpu().numpy())

    result = {}
    real = np.concatenate(real)
    result["real"] = real.squeeze()
    latent = np.concatenate(latent)
    result["latent"] = latent
    reconstruct_signal = np.concatenate(reconstruct_signal)
    result["reconstruct_signal"] = reconstruct_signal.squeeze()
    labels = np.concatenate(labels)
    result["labels"] = labels
    return result
