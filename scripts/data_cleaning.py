import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from utils import *
from data_definition import dataset_structure


class data_cleaning:
    def __init__(self, folder_path):
        self.folder_raw_path = folder_path    
                
    def clean(self, save=True, csv_path=None):
        folder_path = self.folder_raw_path

        dataset = dataset_structure
        csv_file = find_csv_files(directory=folder_path)

        if csv_file:
            df = pd.read_csv(csv_file[0]) 

        for x in range(1,13):
            for y in range(1,7):

                # ------------- VERIFICA IMAGENES -------------- #
                does_exist_seed = True
                mask = {}
                for image_type in ['horizontal','vertical']:        
                    mask_path = os.path.join(folder_path, image_type,f"mask/seedlings_mask_{x}_{y}.jpg")
                    try:
                        mask[image_type] = plt.imread(mask_path)
                    except:
                        does_exist_seed *= does_exist_seed*False
                
                
                #-------------- IMPORTAR DATOS ----------------- #
                
                if does_exist_seed:
                    RAW_DATA = {}
                    for image_type in ['horizontal','vertical']:
                        RAW_DATA[image_type] = from_mask_to_data(mask[image_type])
                    
                    for image_type in ['horizontal', 'vertical']:
                        area = RAW_DATA[image_type]['area']
                        width = RAW_DATA[image_type]['width']
                        height = RAW_DATA[image_type]['height']
                        mask_path = os.path.join(folder_path, image_type,f"mask/seedlings_mask_{x}_{y}.jpg")
                        rgb_path = os.path.join(folder_path, image_type,f"rgb/seedlings_{x}_{y}.jpg")
                        depth_path = os.path.join(folder_path, image_type,f"depth/seedlings_{x}_{y}.jpg")
                             
                        dataset[f'{image_type}_area'].append(area)
                        dataset[f'{image_type}_width'].append(width)
                        dataset[f'{image_type}_height'].append(height)
                        dataset[f'{image_type}_mask_path'].append(mask_path)
                        dataset[f'{image_type}_rgb_path'].append(rgb_path)
                        dataset[f'{image_type}_depth_path'].append(depth_path)

                    length = df[f'col{y}'][f'row{x}']  
                    dataset[f'length'].append(length)
        
        # ----------- GUARDANDO DATOS --------------- #
        dataframe = pd.DataFrame(dataset)
        if save and ~(csv_path is None):
            dataframe.to_csv(csv_path, index=False)
        
        return dataframe



if __name__ == '__main__':
    csv_file_path = './31_03_23.csv'
    cleaner = data_cleaning(
                        folder_path="/home/pqbas/projects/LABINM_Robotics_Automation/SeedlingsNet/classifiers/data/data/raw/31_03_23"
                        )
    
    dataframe = cleaner.clean(save = False)
    print(dataframe)