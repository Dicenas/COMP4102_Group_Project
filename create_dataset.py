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

def apply_noise_reduction(image_path, kernel_size=(5, 5), sigma_x=0):
    """
    Apply Gaussian blur for noise reduction.
    
    Args:
    - image_path (Path): Path to the input image.
    - output_folder (Path): Folder to save the noise-reduced images.
    - kernel_size (tuple): The size of the Gaussian kernel. Larger kernel size means more blurring.
    - sigma_x (float): Standard deviation in X direction for Gaussian kernel. If 0, it is calculated from the kernel size.
    """
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian blur for noise reduction
    smoothed_image = cv2.GaussianBlur(image, kernel_size, sigma_x)
    
    return smoothed_image

def process_images(input_folder, output_folder, select):
    """
    Apply selected correction to all images in the input folder and save the processed images.
    """

    for dir_ in os.listdir(input_folder):
        for img in os.listdir(os.path.join(input_folder, dir_)):
            img_path = os.path.join(input_folder, dir_, img)
            if not os.path.exists(os.path.join(output_folder, dir_)): # Create directory if it doesn't already exist
                os.makedirs(os.path.join(output_folder, dir_))
            
            if select==0:
                corrected_image = apply_lighting_correction(img_path)
            elif select==1:
                corrected_image = apply_noise_reduction(img_path)

            # Save the corrected image
            output_image_path = os.path.join(output_folder, dir_, img)
            
            #plt.figure()
            #plt.imshow(corrected_image, cmap="gray")
            
            success = cv2.imwrite(output_image_path, corrected_image)
            if success:
                print(f"Processed and saved: {output_image_path}")
            else:
                print("Failed to save image.")


DATA_DIR = './data'
LIGHTING_CORRECTION = './preprocessing/lighting'
NOISE_REDUCTION = './preprocessing/noise'
process_images(DATA_DIR, LIGHTING_CORRECTION, 0)
process_images(LIGHTING_CORRECTION, NOISE_REDUCTION, 1)

#plt.show()



