import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
import os.path
import shutil
import uuid
from utils import *
import pandas


df = pd.read_csv("/home/pqbas/projects/LABINM_Robotics_Automation/SeedlingsNet/classifiers/data/data/raw/31_03_23/alturas.csv")

n_cells_wihtout_seedlings = (df[:][:] == -1).sum().sum()
total_cells = df.size

print('% percentage_cells:', ((total_cells - n_cells_wihtout_seedlings)/total_cells)*100)


folders_to_include = [
    "03_03_23",
    #"03_03_23_2",
    #"17_03_23",
    #"24_03_23",
    #"31_03_23"
]


dataset = {
    'horizontal_area': [],
    'vertical_area': [],
    
    'horizontal_height': [],
    'vertical_height': [],
    
    'horizontal_width': [],
    'vertical_width': [],

    'horizontal_mask_path': [],
    'vertical_mask_path': [],

    'horizontal_rgb_path': [],
    'vertical_rgb_path': [],

    'horizontal_depth_path': [],
    'vertical_depth_path': []
    #'length': []
}


gallery_path = [
    "03_03_23",
    "03_03_23_2",
    "17_03_23",
    "24_03_23",
    "31_03_23",
]


folder_path = "/home/pqbas/projects/LABINM_Robotics_Automation/SeedlingsNet/classifiers/data/data/raw/31_03_23" #"dataset"

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


# ----------- GUARDANDO DATOS --------------- #
dataframe = pandas.DataFrame(dataset)
dataframe.to_csv('sample.csv', index=False)
dataframe



#         
        
#         # ------------- IMPORTA LOS DATOS -------------- #
#         for image_type in ['horizontal','vertical']:        
#             image_path = os.path.join(folder_path, image_type,f"mask/seedlings_mask_{x}_{y}.jpg")
#            
#             print(image)
#             raw_data_i = 
#             if raw_data_i  == None:
#                 next
      
#         print(RAW_DATA)
#         for image_type in ['horizontal','vertical']:
#             data_from_projection = RAW_DATA[image_type]
#             print(RAW_DATA)
#             print(data_from_projection)
#             
#             dataset[f'{image_type}_width'].append(data_from_projection['width'])
#             dataset[f'{image_type}_heigth'].append(data_from_projection['heigth'])
            
#             #dataset[f'{image_type}_mask_path'].append(RAW_DATA)


# dataframe = pandas.DataFrame(dataset)
# dataframe.to_csv('sample.csv', index=False)
# dataframe
        

# dataset = almacenar_datos(dataset, df, folder_path, x, y)


# # ------------- ALMACENAR LOS DATOS -------------- #
#



# print(df)



#         print(f"Archivos encontrados para {folder}/{x}_{y}")
#         almacenar_datos(folder_path, x, y)



#for folder in folders_to_include:
#    folder_path = f"../{folder}/"
#    print(folder_path)

#     csv_file = find_csv_files(folder_path)

#     if csv_file:
#         df = pd.read_csv(csv_file[0])   

#     for x in range(1,13):
#         for y in range(1,7):
#             if verificar_images(folder_path, x, y) is False:
#                 print(f"No se encontraron archivos para {folder}/{x}_{y}")
#                 continue

#             print(f"Archivos encontrados para {folder}/{x}_{y}")
#             almacenar_datos(folder_path, x, y)
# copy_image(gallery_path, dataset_path)

"""
Este script crea una mascara de segmentacion del plantin de alcachofa en la cual calcula la longitud de sus 
hojas, su area, altura y el ancho de la region del plantin, esta informacion la organiza en un formato tabular.
Tambien realiza copias de archivos de imagenes de una ubicacion a otra y les asigna nombres unicos.
"""


