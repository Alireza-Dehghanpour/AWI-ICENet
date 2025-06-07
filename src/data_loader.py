# author: Alireza Dehghanpour

# --------------------------------------------------------------------------------
# Libraries Import 
# RGV2ZWxvcGVkIGJ5IEFsaXJlemEgRGVoZ2hhbnBvdXIsIDIwMjMsIGZvciBtb3JlIGluZm8gcGxlYXNlIGNvbnRhY3QgYS5yLmRlaGdoYW5wb3VyQGdtYWlsLmNvbQ==


import os
import numpy as np
from tqdm import tqdm
import scipy.interpolate
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# Utility Functions

# Normalize a noise waveform by its maximum amplitude
def normal_noise(lst):
    m = max(lst)
    for i in range(len(lst)):
        lst[i] /= m



# --------------------------------------------------------------------------------
# Data Loading Pipeline
# NOTE: This section handles binary file reading and array reshaping 
def load_data(path):
    ATT_lst, Sim_NO_noise_lst, SIM_WF_NOISE_lst, OCOG_lst, OCOG_NOISE_lst, REF_lst = [], [], [], [], [], []
    lst_file = os.listdir(path)
    lst_file.sort()
    lst_number = []
    
    for item_name in tqdm(lst_file):
        if "ATT" in item_name:
            idx_file = item_name.split("_")[-1].split(".")[0]
            lst_number.append(idx_file)

            ATT_lst.append(np.fromfile(os.path.join(path, f"ATT_ARR_{idx_file}.bin"), np.double))
            Sim_NO_noise_lst.append(np.fromfile(os.path.join(path, f"SIM_WF_NO_NOISE_{idx_file}.bin"), np.float32).reshape(-1, 128))
            temp_noise = np.fromfile(os.path.join(path, f"SIM_WF_NOISE_{idx_file}.bin"), np.float32).reshape(-1, 10, 128)
            SIM_WF_NOISE_lst.append(temp_noise)
            OCOG_lst.append(np.fromfile(os.path.join(path, f"OCOG_ARR_{idx_file}.bin"), np.double))
            REF_lst.append(np.fromfile(os.path.join(path, f"REF_ARR_{idx_file}.bin"), np.double))
            ocog_noise = np.fromfile(os.path.join(path, f"OCOG_ARR_NOISE_{idx_file}.bin"), np.double).reshape(-1, 10)
            OCOG_NOISE_lst.append(ocog_noise)


# Return all loaded data as numpy arrays

    return (
        np.array(ATT_lst),
        np.array(Sim_NO_noise_lst),
        np.array(SIM_WF_NOISE_lst),
        np.array(OCOG_lst),
        np.array(OCOG_NOISE_lst),
        np.array(REF_lst),
        np.array(lst_number)
    )



# --------------------------------------------------------------------------------
# Data Extraction for Model Training


def extract_data(SIM_WF_NOISE_lst, OCOG_lst, REF_lst):
    SIM_WF_NOISE_lst = SIM_WF_NOISE_lst[0] 
    OCOG_lst = OCOG_lst[0]  
    REF_lst = REF_lst[0]    

    OCOG_lst = np.repeat(OCOG_lst, 5)
    REF_lst = np.repeat(REF_lst, 5)

    data_wave = []
    lbl_OCOG = []
    lbl_REF = []

    n_profiles = SIM_WF_NOISE_lst.shape[0]  
    n_noises = SIM_WF_NOISE_lst.shape[1]   

    print("_" * 100)
    print(f"✓  Total profiles: {n_profiles}, noises per profile: {n_noises}")
    print("_" * 100)

    for i in range(n_profiles):  
        for j in range(n_noises): 
            data_wave.append(SIM_WF_NOISE_lst[i, j])
            lbl_OCOG.append(OCOG_lst[i])
            lbl_REF.append(REF_lst[i])

    data_wave = np.array(data_wave)
    lbl_REF = np.array(lbl_REF)

    print("_" * 100)
    print(f"✓  Final data_wave shape: {data_wave.shape}")
    print(f"✓  Final lbl_REF shape: {lbl_REF.shape}")
    print("_" * 100)

    return data_wave, lbl_REF
# RGV2ZWxvcGVkIGJ5IEFsaXJlemEgRGVoZ2hhbnBvdXIsIDIwMjMsIGZvciBtb3JlIGluZm8gcGxlYXNlIGNvbnRhY3QgYS5yLmRlaGdoYW5wb3VyQGdtYWlsLmNvbQ==
