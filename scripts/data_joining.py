import pandas as pd
import matplotlib.pyplot as plt
import os
import uuid


class data_joining:
    def __init__(self, dataseed_path):
        self.dataseed = pd.read_csv(dataseed_path)

    def join(self, dataframe_path):
        dataframe = pd.read_csv(dataframe_path)
        
        for index, row in dataframe.iterrows():
            horizontal_height = row['horizontal_height']
            horizontal_width = row['horizontal_width']
            horizontal_area = row['horizontal_area']
            horizontal_rgb_path = row['horizontal_rgb_path']
            horizontal_mask_path = row['horizontal_mask_path']
            horizontal_depth_path = row['horizontal_depth_path']

            vertical_height = row['vertical_height']
            vertical_width = row['vertical_width']
            vertical_area = row['vertical_area']
            vertical_rgb_path = row['vertical_rgb_path']
            vertical_mask_path = row['vertical_mask_path']
            vertical_depth_path = row['vertical_depth_path']

            length = row['length']

    

            new_row = {
                "horizontal_height":horizontal_height,
                "horizontal_width":horizontal_width,
                "horizontal_area":horizontal_area,
                "horizontal_rgb_path":horizontal_rgb_path,
                "horizontal_mask_path":horizontal_mask_path,
                "horizontal_depth_path":horizontal_depth_path,
                "vertical_height":vertical_height,
                "vertical_width":vertical_width,
                "vertical_area":vertical_area,
                "vertical_rgb_path":vertical_rgb_path,
                "vertical_mask_path":vertical_mask_path,
                "vertical_depth_path":vertical_depth_path,
                "length":length
            }

            self.dataseed = pd.concat([self.dataseed, pd.DataFrame([new_row])], ignore_index=True)
        return
    
    def get_dataseed(self):
        return self.dataseed
    
    def save_dataseed(self, csv_path):
        self.dataseed.to_csv(csv_path, index=False)
        return


if __name__ == '__main__':
    
    list_dataframe_path_ = [
        '/home/pqbas/projects/LABINM_Robotics_Automation/SeedlingsNet/classifiers/data/data/processed/03_03_23.csv',
        '/home/pqbas/projects/LABINM_Robotics_Automation/SeedlingsNet/classifiers/data/data/processed/17_03_23.csv',
        '/home/pqbas/projects/LABINM_Robotics_Automation/SeedlingsNet/classifiers/data/data/processed/24_03_23.csv',
        '/home/pqbas/projects/LABINM_Robotics_Automation/SeedlingsNet/classifiers/data/data/processed/31_03_23.csv'
    ]
    
    dataseed_path_ = '/home/pqbas/projects/LABINM_Robotics_Automation/SeedlingsNet/classifiers/data/dataset/data.csv'
    

    joiner = data_joining(dataseed_path = dataseed_path_)
    for dataframe_path_ in list_dataframe_path_:
        joiner.join(dataframe_path_)
    
    dataseed_new = joiner.get_dataseed()

    joiner.save_dataseed('/home/pqbas/projects/LABINM_Robotics_Automation/SeedlingsNet/classifiers/data/dataset/data.csv')
    print(dataseed_new)
    
