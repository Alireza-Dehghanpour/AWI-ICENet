# author: Alireza Dehghanpour

# --------------------------------------------------------------------------------
# Libraries Import 
# RGV2ZWxvcGVkIGJ5IEFsaXJlemEgRGVoZ2hhbnBvdXIsIDIwMjMsIGZvciBtb3JlIGluZm8gcGxlYXNlIGNvbnRhY3QgYS5yLmRlaGdoYW5wb3VyQGdtYWlsLmNvbQ==


import os
import numpy as np
from sklearn.model_selection import KFold
from tensorflow import keras
import pandas as pd
from data_loader import extract_data, load_data
from model import create_cnn_model
from utils import get_metrics, init_results_dataframe


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# Model Training Pipeline


def train_model(path, cv_folds=5, batch_size=128, epochs=30, input_size=(128, 1), learning_rate=2e-4):
    ATT_lst, Sim_NO_noise_lst, SIM_WF_NOISE_lst, OCOG_lst, OCOG_NOISE_lst, REF_lst, lst_number = load_data(path)

    ATT_lst = np.array(ATT_lst)
    Sim_NO_noise_lst = np.array(Sim_NO_noise_lst)
    SIM_WF_NOISE_lst = np.array(SIM_WF_NOISE_lst)
    OCOG_lst = np.array(OCOG_lst)
    OCOG_NOISE_lst = np.array(OCOG_NOISE_lst)
    REF_lst = np.array(REF_lst)
    # Data shapes — for validation and debugging

    print("       Data.shape ")
    print("_" * 100)
    print(f"  ✓    ATT_lst: {ATT_lst.shape}")
    print(f"  ✓    Sim_NO_noise_lst: {Sim_NO_noise_lst.shape}")
    print(f"  ✓    SIM_WF_NOISE_lst: {SIM_WF_NOISE_lst.shape}")
    print(f"  ✓    OCOG_lst: {OCOG_lst.shape}")
    print(f"  ✓    OCOG_NOISE_lst: {OCOG_NOISE_lst.shape}")
    print(f"  ✓    REF_lst: {REF_lst.shape}")
    print("_" * 100)

    data_wave, lbl_REF = extract_data(SIM_WF_NOISE_lst, OCOG_lst, REF_lst)

    algorithm_name = ["CNN"]
    res_metrics_train = init_results_dataframe(algorithm_name, cv_folds)
    res_metrics_test = res_metrics_train.copy()

    kf = KFold(n_splits=cv_folds, shuffle=False)
    fold = 0
    row = -1

    for train_idx, test_idx in kf.split(data_wave):
        fold += 1
        row += 1

        train_data, test_data = data_wave[train_idx], data_wave[test_idx]
        train_lbl, test_lbl = lbl_REF[train_idx], lbl_REF[test_idx]


        train_data = train_data.reshape((train_data.shape[0], train_data.shape[1], 1))
        test_data = test_data.reshape((test_data.shape[0], test_data.shape[1], 1))

        print("_" * 100)
        print(f"TRAIN: {train_data.shape[0]} TEST: {test_data.shape[0]}")
        print(f"CNN in fold: {fold}")

        print("       Data.shape ")
        print("_"*100)

        model = create_cnn_model(input_size=input_size, learning_rate=learning_rate)
        model.summary()
        
        # Train the model — with validation split

        history = model.fit(
            train_data, train_lbl,
            validation_data=(test_data, test_lbl),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=1
        )

# --------------------------------------------------------------------------------


        pred_train = model.predict(train_data, batch_size=batch_size)
        pred_test = model.predict(test_data, batch_size=batch_size)


        res_metrics_train = get_metrics(train_lbl, pred_train, res_metrics_train, row, fold)
        res_metrics_test = get_metrics(test_lbl, pred_test, res_metrics_test, row, fold)

        model_filename = os.path.join(path, f'DCNN_model_fold{fold}.h5')
        history_filename = os.path.join(path, f'history_fold{fold}.npy')

        model.save(model_filename)
        np.save(history_filename, history.history)


    res_metrics_train.to_csv(os.path.join(path, "train_metrics.csv"), index=False)
    res_metrics_test.to_csv(os.path.join(path, "test_metrics.csv"), index=False)

    print("Training completed. Metrics saved!")
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    path = r"C:\Users\ardeh\OneDrive\Desktop\awi-icenet1-retracker\data"  
    train_model(path)

# RGV2ZWxvcGVkIGJ5IEFsaXJlemEgRGVoZ2hhbnBvdXIsIDIwMjMsIGZvciBtb3JlIGluZm8gcGxlYXNlIGNvbnRhY3QgYS5yLmRlaGdoYW5wb3VyQGdtYWlsLmNvbQ==
