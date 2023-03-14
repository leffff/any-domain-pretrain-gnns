import random

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold


# splitter function


def random_split(dataset, task_idx=None, null_value=0,
                 frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=0,
                 smiles_list=None):
    """

    :param dataset:
    :param task_idx:
    :param null_value:
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param seed:
    :param smiles_list: list of smiles corresponding to the dataset obj, or None
    :return: train, valid, test slices of the input dataset obj. If
    smiles_list != None, also returns ([train_smiles_list],
    [valid_smiles_list], [test_smiles_list])
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx != None:
        # filter based on null values in task_idx
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        non_null = y_task != null_value  # boolean array that correspond to non null values
        idx_array = np.where(non_null)[0]
        dataset = dataset[torch.tensor(idx_array)]  # examples containing non
        # null labels in the specified task_idx
    else:
        pass

    num_mols = len(dataset)
    random.seed(seed)
    all_idx = list(range(num_mols))
    random.shuffle(all_idx)

    train_idx = all_idx[:int(frac_train * num_mols)]
    valid_idx = all_idx[int(frac_train * num_mols):int(frac_valid * num_mols)
                                                   + int(frac_train * num_mols)]
    test_idx = all_idx[int(frac_valid * num_mols) + int(frac_train * num_mols):]

    assert len(set(train_idx).intersection(set(valid_idx))) == 0
    assert len(set(valid_idx).intersection(set(test_idx))) == 0
    assert len(train_idx) + len(valid_idx) + len(test_idx) == num_mols

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset = dataset[torch.tensor(test_idx)]

    if not smiles_list:
        return train_dataset, valid_dataset, test_dataset
    else:
        train_smiles = [smiles_list[i] for i in train_idx]
        valid_smiles = [smiles_list[i] for i in valid_idx]
        test_smiles = [smiles_list[i] for i in test_idx]
        return train_dataset, valid_dataset, test_dataset, (train_smiles,
                                                            valid_smiles,
                                                            test_smiles)


def cv_random_split(dataset, fold_idx=0,
                    frac_train=0.9, frac_valid=0.1, seed=0,
                    smiles_list=None):
    """

    :param dataset:
    :param task_idx:
    :param null_value:
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param seed:
    :param smiles_list: list of smiles corresponding to the dataset obj, or None
    :return: train, valid, test slices of the input dataset obj. If
    smiles_list != None, also returns ([train_smiles_list],
    [valid_smiles_list], [test_smiles_list])
    """

    np.testing.assert_almost_equal(frac_train + frac_valid, 1.0)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

    labels = [data.y.item() for data in dataset]

    idx_list = []

    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, val_idx = idx_list[fold_idx]

    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(val_idx)]

    return train_dataset, valid_dataset
