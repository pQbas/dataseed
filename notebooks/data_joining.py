import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import os
import os.path
import shutil
import uuid

df = pd.read_csv("/home/bruno29/catkin_ws/dataseed/31_03_23/Plantines_31_03_23 - Sheet1.csv")

n_cells_wihtout_seedlings = (df[:][:] == -1).sum().sum()
total_cells = df.size

print('% percentage_cells:', ((total_cells - n_cells_wihtout_seedlings)/total_cells)*100)

df

folders_to_include = [
    "03_03_23",
    "03_03_23_2",
    "17_03_23",
    "24_03_23",
    "31_03_23"
]

dataset = {
    'vertical_area': [],
    'vertical_heigth': [],
    'vertical_width': [],
    'horizontal_area': [],
    'horizontal_heigth': [],
    'horizontal_width': [],
    'vertical_mask_path': [],
    'horizontal_mask_path': [],
    'vertical_rgb_path': [],
    'horizontal_rgb_path': [],
    'vertical_depth_path': [],
    'horizontal_depth_path': [],
    'length': []
}

def get_bbox_from_mask(mask):

    seg_value = 1

    if mask is not None:
        np_seg = np.array(mask)
        segmentation = np.where(np_seg == seg_value)

        # Bounding Box
        bbox = 0, 0, 0, 0
        if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
            x_min = int(np.min(segmentation[1]))
            x_max = int(np.max(segmentation[1]))
            y_min = int(np.min(segmentation[0]))
            y_max = int(np.max(segmentation[0]))
            bbox = x_min, y_min, x_max, y_max
            return bbox
        
        return None
    else:
        # Handle error case where segmentation image cannot be read or is empty
        print("Error: Segmentation image could not be read or is empty.")
        return None


def get_data_from_gallery(path):

    try:
        mask = plt.imread(path)
    except FileNotFoundError:
        print(f"Warning: File not found - {path}")
        return None

    x_min, y_min, x_max, y_max = get_bbox_from_mask(mask)
    if x_min is None or y_min is None or x_max is None or y_max is None:
        # Si no se puede calcular la bounding box, retorna None
        print(f"Warning: Unable to calculate bounding box for - {path}")
        return None
    
    height = y_max - y_min
    width = x_max - x_min
    area = cv2.countNonZero(mask)

    return {
        'heigth':height,
        'width':width,
        'area':area,
    }

def find_csv_files(directory):
    """
    # Specify the directory you want to search in
    directory_to_search = "../gallery_03_03_23_tray1"
    
    # Call the function to find CSV files in the directory
    csv_files_found = find_csv_files(directory_to_search) 

    if csv_files_found:
        print("CSV files found:")
        for csv_file in csv_files_found:
            print(csv_file)
    else:
        print("No CSV files found in the directory:", directory_to_search)

    """
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files

def verificar_images(folder_path, x, y):
    for image_type in ['vertical', 'horizontal']:
        image_path = f"{folder_path}{image_type}/mask/seedlings_mask_{x}_{y}.jpg"
        
        try:
            ver = get_data_from_gallery(image_path)
            return ver
        
        except FileNotFoundError:
            continue

def copy_and_rename(src_folder, dst_folder):
    if not os.path.exists(dst_folder):
        print(f"Error: La carpeta de destino {dst_folder} no existe.")
        exit()

    files = [f for f in os.listdir(src_folder) if f.endswith(".jpg")]
    for filename in files:
        src = os.path.join(src_folder, filename)
        unique_name = f"{str(uuid.uuid4())[:8]}.jpg"
        dst = os.path.join(dst_folder, unique_name)
        shutil.copy(src, dst)
        return unique_name

def copy_image(gallery_path, dataset_path):
    for folder_name in gallery_path:
        folder_total = f"{folder_name}"

        ruta_origen_A = f"/home/bruno29/catkin_ws/dataseed/{folder_total}/horizontal/rgb"
        ruta_origen_B = f"/home/bruno29/catkin_ws/dataseed/{folder_total}/vertical/rgb"

        ruta_destino_C = f"/home/bruno29/catkin_ws/dataseed/dataset/horizontal/rgb"
        ruta_destino_D = f"/home/bruno29/catkin_ws/dataseed/dataset/vertical/rgb"

        print(f"Copying and renaming from {ruta_origen_A} to {ruta_destino_C}...")
        copy_and_rename(ruta_origen_A, ruta_destino_C)

        print(f"Copying and renaming from {ruta_origen_B} to {ruta_destino_D}...")
        copy_and_rename(ruta_origen_B, ruta_destino_D)

    print("Completed Process.")            

def almacenar_datos(folder_path, x, y):
    for image_type in ['vertical', 'horizontal']:
        path = f"{folder_path}{image_type}/mask/seedlings_mask_{x}_{y}.jpg"
        data = get_data_from_gallery(path)
        
        if data:
            dataset[f'{image_type}_area'].append(data['area'])
            dataset[f'{image_type}_width'].append(data['width'])
            dataset[f'{image_type}_heigth'].append(data['heigth'])
            dataset[f'{image_type}_mask_path'].append(path)
            
            unique_name = copy_and_rename(
                os.path.join(folder_path, f"{image_type}/rgb"),
                os.path.join("../dataset/", f"{image_type}/rgb")
            )

            rgb_path = os.path.join("../dataset/", f"{image_type}/rgb/{unique_name}")
            dataset[f'{image_type}_rgb_path'].append(rgb_path)

        path = f"{folder_path}{image_type}/depth/seedlings_{x}_{y}.jpg"  
        if data:
            dataset[f'{image_type}_depth_path'].append(path)


    if data != None:
        length = df[f'col{y}'][f'row{x}']
        dataset[f'length'].append(length)

gallery_path = [
    "03_03_23",
    "03_03_23_2",
    "17_03_23",
    "24_03_23",
    "31_03_23",
]

dataset_path = "dataset"

for folder in folders_to_include:
    folder_path = f"../{folder}/"

    csv_file = find_csv_files(folder_path)

    if csv_file:
        df = pd.read_csv(csv_file[0])   

    for x in range(1,13):
        for y in range(1,7):
            if verificar_images(folder_path, x, y) is False:
                print(f"No se encontraron archivos para {folder}/{x}_{y}")
                continue

            print(f"Archivos encontrados para {folder}/{x}_{y}")
            almacenar_datos(folder_path, x, y)
copy_image(gallery_path, dataset_path)

"""
Este script crea una mascara de segmentacion del plantin de alcachofa en la cual calcula la longitud de sus 
hojas, su area, altura y el ancho de la region del plantin, esta informacion la organiza en un formato tabular.
Tambien realiza copias de archivos de imagenes de una ubicacion a otra y les asigna nombres unicos.
"""


