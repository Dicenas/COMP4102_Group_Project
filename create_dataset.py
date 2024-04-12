import os

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np

def apply_lighting_correction(image_path):
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply histogram equalization
    corrected_image = cv2.equalizeHist(image)
    
    return corrected_image

def apply_noise_reduction(image_path, kernel_size=(5, 5), sigma_x=0):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian blur for noise reduction
    smoothed_image = cv2.GaussianBlur(image, kernel_size, sigma_x)
    
    return smoothed_image

def apply_background_segmentation(image_path):
    # Load the image in grayscale (assuming it's already in grayscale from previous steps)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Otsu's thresholding; returns (threshold used by the function which is computed automatically, thresholded image)
    _, segmented_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    

    #segmented_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #segmented_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    return segmented_image

def apply_edge_detection(image_path):
    # Load the image and convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Edge Detection
    edges = cv2.Canny(gray, 25, 50)

    # Step 2: Morphological Operations - Dilation
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Step 3: Find and Fill Contours
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hand_mask = np.zeros_like(gray)
    cv2.drawContours(hand_mask, contours, -1, color=255, thickness=cv2.FILLED)

    # Step 4: Apply the Mask
    segmented_hand = cv2.bitwise_and(image, image, mask=hand_mask)

    return segmented_hand
    
def resize_image(image_path, target_size=(48, 48)):
    # Load the image
    image = cv2.imread(image_path)
    
    # Resize the image
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    return resized_image

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
            elif select==2:
                corrected_image = apply_background_segmentation(img_path)
            elif select==3:
                corrected_image = resize_image(img_path)
            elif select==4:
                corrected_image = apply_edge_detection(img_path)

            # Save the corrected image
            output_image_path = os.path.join(output_folder, dir_, img)
            
            #plt.figure()
            #plt.imshow(corrected_image, cmap="gray")
            
            success = cv2.imwrite(output_image_path, corrected_image)
            if success:
                print(f"Processed and saved: {output_image_path}")
            else:
                print("Failed to save image.")


DATA_DIR = './background_data'
LIGHTING_CORRECTION = './preprocessing/lighting'
NOISE_REDUCTION = './preprocessing/noise'
BACKGROUND_SEGMENTATION = './preprocessing/background'
RESIZE_IMAGE = './preprocessing/resize'


process_images(DATA_DIR, LIGHTING_CORRECTION, 0)
process_images(LIGHTING_CORRECTION, NOISE_REDUCTION, 1)
process_images(NOISE_REDUCTION, BACKGROUND_SEGMENTATION, 2)
process_images(BACKGROUND_SEGMENTATION, RESIZE_IMAGE, 3)

#process_images(DATA_DIR, './preprocessing/adaptive_gaussian_thresholding', 2)
#process_images(DATA_DIR, './preprocessing/adaptive_mean_thresholding', 2)

CANNY_SEGMENTATION = './preprocessing/canny'
process_images(DATA_DIR, CANNY_SEGMENTATION, 4)
#plt.show()



