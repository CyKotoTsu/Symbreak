import pandas as pd
import pickle
import numpy as np

def cell_door_trans(file, save_path):
    '''
    Docstring for cell_door_trans
    
    :param file: Path to the input .npy file
    :param save_path: Path for the output csv with the correct format for cellular door
    '''

    with open(file , 'rb') as f:
        p_mask_lst, x_lst, p_lst, q_lst, U_lst = pickle.load(f)

    # Row 1: cells per frame
    cell_ct_lst = []
    for i in range(len(x_lst)):
        cell_ct_lst.append(len(x_lst[i]))

    # Row 2: Cell type explanations
    cell_types = ['0: Epiblast', '1: emVE', '2: DVE']

    # Row 3: Define the scalars
    scalars = ['scalars: DKK', 'Wnt']

    # Row 4 and onwards: Cell positions and scalar values
    data_rows = []
    for frame in range(len(x_lst)):
        frame_data = []
        for cell in range(len(x_lst[frame])):
            cell_info = [
                x_lst[frame][cell][0].item(),  # x position
                x_lst[frame][cell][1].item(),  # y position
                x_lst[frame][cell][2].item(),  # z position
                p_mask_lst[frame][cell].item(), # Cell type
                U_lst[frame][cell][1].item(),   # DKK scalar value
                U_lst[frame][cell][2].item()    # Wnt scalar value
            ]
            frame_data.append(cell_info)
        data_rows.append(frame_data)

     # Write to CSV separated by commas
    with open(save_path, 'w') as f:
        # Write the first three rows
        f.write(','.join(map(str, cell_ct_lst)) + '\n')
        f.write(','.join(cell_types) + '\n')
        f.write(','.join(scalars) + '\n')

        # Write the cell data for each frame
        for frame_data in data_rows:
            for cell_info in frame_data:
                f.write(','.join(map(str, cell_info)) + '\n')
    return()
            