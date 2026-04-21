import numpy as np
import pandas as pd
import pickle


# Custom time frame to df
def npy_to_csv(file,output_csv,frame=None):
    '''
    Convert a specific time frame data to csv for plotting in other software

    Input:
        file (str): path to .npy file containing data on all time points
        output_csv (str): path to output csv file
        frame (int): the time frame to convert if not None, otherwise convert last time point
    Returns:
        csv file of last time point data
    '''


    with open(file,'rb') as f:
        p_mask_lst, x_lst, p_lst, q_lst, U_lst = pickle.load(f)



    # Take last time point arrays
    p_mask = p_mask_lst[frame] if frame is not None else p_mask_lst[-1]
    x = x_lst[frame] if frame is not None else x_lst[-1]
    U = U_lst[frame] if frame is not None else U_lst[-1]

    # Create dataframe
    df = pd.DataFrame({
        'x': x[:,0],
        'y': x[:,1],
        'z': x[:,2],
        'p_mask': p_mask,
        'BMP': U[:,0],
        'DKK': U[:,1],
        'Wnt': U[:,2]
    })

    # Save to csv
    df.to_csv(output_csv, index=False)
    return