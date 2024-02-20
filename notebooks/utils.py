import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import shutil


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
        #print(f"Warning: File not found - {path}")
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
        'height':height,
        'width':width,
        'area':area,
    }

def from_mask_to_data(mask):
    x_min, y_min, x_max, y_max = get_bbox_from_mask(mask)
    if x_min is None or y_min is None or x_max is None or y_max is None:
        # Si no se puede calcular la bounding box, retorna None
        print(f"Warning: Unable to calculate bounding box for - {path}")
        return None
    
    height = y_max - y_min
    width = x_max - x_min
    area = cv2.countNonZero(mask)
    
    return {
        'height':height,
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


def does_exist_horizontal_vertical_images(folder_path, x, y):

    for image_type in ['vertical', 'horizontal']:        
        image_path = os.path.join(folder_path, image_type,f"mask/seedlings_mask_{x}_{y}.jpg")

        try:
            get_data_from_gallery(image_path)
        except:
            return False

    return True

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


def almacenar_datos(dataset, df, folder_path, x, y):
    for image_type in ['vertical', 'horizontal']:
        path = os.path.join(folder_path, image_type, "mask/seedlings_mask_{x}_{y}.jpg")
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

    return dataset