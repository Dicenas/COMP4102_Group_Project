import os

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

def apply_lighting_correction(image_path):
    """
    Apply histogram equalization to an image for lighting correction.
    Args:
    - image_path (str or Path): The path to the input image.
    
    Returns:
    - corrected_image: The lighting-corrected image.
    """
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply histogram equalization
    corrected_image = cv2.equalizeHist(image)
    
    return corrected_image

def process_dataset(dataset_folder, output_folder):
    """
    Apply lighting correction to all images in the dataset folder and save the processed images.
    Args:
    - dataset_folder (str or Path): The folder containing the dataset images.
    - output_folder (str or Path): The folder where corrected images will be saved.
    """

    for dir_ in os.listdir(dataset_folder):
        for img in os.listdir(os.path.join(dataset_folder, dir_)):
            img_path = os.path.join(dataset_folder, dir_, img)

            if not os.path.exists(os.path.join(output_folder, dir_)): # Create directory if it doesn't already exist
                os.makedirs(os.path.join(output_folder, dir_))
            
            corrected_image = apply_lighting_correction(img_path)

            # Save the corrected image
            output_image_path = os.path.join(output_folder, dir_, img)
            success = cv2.imwrite(output_image_path, corrected_image)
            if success:
                print(f"Processed and saved: {output_image_path}")
            else:
                print("Failed to save image.")



DATA_DIR = './data'
LIGHTING_CORRECTION = './preprocessing/lighting'
process_dataset(DATA_DIR, LIGHTING_CORRECTION)



