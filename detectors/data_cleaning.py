import cv2
import os
from yolo7 import Yolo7
import sys

sys.path.insert(1, 'yolov7')
sys.path.append('/home/bruno29/catkin_ws/dataseed/detectors')
sys.path.append('/home/bruno29/catkin_ws/dataseed/detectors/yolov7')
sys.path.append('/home/bruno29/catkin_ws/dataseed/detectors/yolov7/utils')
sys.path.append('/home/bruno29/miniconda3/envs/dl/lib/python3.8/site-packages')

#from yolov7.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages,LoadStreams, PassImage

# Creando las instancias de Yolo7
detector_horizontal = Yolo7(weights='/home/bruno29/catkin_ws/dataseed/detectors/weights/yolov7-hseed.pt',
                            data='./detectors/opt.yaml', 
                            device='cuda:0')

detector_vertical = Yolo7(weights='/home/bruno29/catkin_ws/dataseed/detectors/weights/yolov7-vseed.pt',
                          data='./detectors/opt.yaml', 
                          device='cuda:0')


def h_mask(folder_path, x, y):
    try:
        src = f"/home/bruno29/catkin_ws/dataseed/{folder_path}/horizontal/rgb"
        img = cv2.imread(f"{src}/seedlings_{x}_{y}.jpg")
        predictions = detector_horizontal.predict(img)
    except FileNotFoundError:
        return 
    
    if predictions is not None:
        result = detector_horizontal.plot_prediction(img, predictions)
        mascara = predictions[0].mask
        mascara_gray = cv2.normalize(mascara, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Copiar imágenes
        dst = f"/home/bruno29/catkin_ws/dataseed/{folder_path}/horizontal/mask"
        mask_filename = f"seedlings_mask_{x}_{y}.jpg"
        mask_os = os.path.join(dst, mask_filename)
        cv2.imwrite(mask_os, mascara_gray)
        print(f"Se guardó la imagen {dst}/seedlings_{x}_{y} en la carpeta /horizontal/mask")

def v_mask(folder_path, x, y):
    try:
        src = f"/home/bruno29/catkin_ws/dataseed/{folder_path}/vertical/rgb"
        img = cv2.imread(f"{src}/seedlings_{x}_{y}.jpg")
        predictions = detector_vertical.predict(img)
    except FileNotFoundError:
        return 

    if predictions is not None:
        result = detector_vertical.plot_prediction(img, predictions)
        mascara = predictions[0].mask
        mascara_gray = cv2.normalize(mascara, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Copiar imágenes
        dst = f"/home/bruno29/catkin_ws/dataseed/{folder_path}/vertical/mask"
        mask_filename = f"seedlings_mask_{x}_{y}.jpg"
        mask_os = os.path.join(dst, mask_filename)
        cv2.imwrite(mask_os, mascara_gray)
        print(f"Se guardó la imagen {dst}/seedlings_{x}_{y} en la carpeta /vertical/mask")

if __name__ == "__main__":
    
    """
    Este script utiliza nuestro modelo de yolov7 para realizar la deteccion 
    de plantines de alcachofa en dos planos diferentes (horizontal y vertical), 
    despues obtiene y guarda las mascaras resultantes de cada plantin en carpetas especificas
    """

    folders_to_include = [
        "07_07_23",
        "09_06_23",
        "19_05_23",
        "21_04_23",
    ]
    for folder in folders_to_include:
        folder_path = f"{folder}"
        
        for x in range(1, 13):
            for y in range(1, 7):
                h_mask(folder_path, x, y)
                v_mask(folder_path, x, y)

