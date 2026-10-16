"""Hold out each digit class in turn as the anomaly; compare autoencoder and PCA by AUC."""

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from autoencoder import pca_error, reconstruction_error, train

torch.set_num_threads(4)
images, labels = load_digits(return_X_y=True)
images = images / 16.0                                    # pixel values 0..16 -> 0..1
print('held-out class  PCA AUC  autoencoder AUC (mean, min over seeds 0-2)')
pca_all, ae_all = [], []
for anomaly in range(10):
    normal, odd = images[labels != anomaly], images[labels == anomaly]
    train_x, test_normal = train_test_split(normal, test_size=0.3, random_state=0)
    test_x = np.vstack([test_normal, odd])
    is_anomaly = np.r_[np.zeros(len(test_normal)), np.ones(len(odd))]
    pca = PCA(n_components=8, random_state=0).fit(train_x)
    pca_auc = roc_auc_score(is_anomaly, pca_error(pca, test_x))
    ae_auc = [roc_auc_score(is_anomaly, reconstruction_error(train(train_x, seed), test_x))
              for seed in (0, 1, 2)]
    pca_all.append(pca_auc)
    ae_all.append(np.mean(ae_auc))
    print(f'{anomaly:14d}  {pca_auc:7.3f}  {np.mean(ae_auc):.3f}, {min(ae_auc):.3f}')
print(f'mean over classes: PCA {np.mean(pca_all):.3f}, autoencoder {np.mean(ae_all):.3f}')
