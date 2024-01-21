# -----------------------------------------------------------------------------
# Preprocessing Class
# Author: Xavier Beltran Urbano and Zain Muhammad
# Date Created: 17-11-2023
# -----------------------------------------------------------------------------

# Import libraries
import numpy as np
from network import SegNetwork
import cv2


# Preprocessing Class
class Preprocessing:
    
    def __init__(self,target_size):
        # Export Model
        network = SegNetwork()
        model = network.exportModel()
        model.load_weights("/notebooks/Model_segmentation/Best_weights.h5")
        self.modelSegmentation=model
        self.target_size=target_size

    def extractROI(self,img, threshold=50):
        # image dimensions
        h, w = img.shape[:2]

        # coordinates of the pixels in the diagonal
        y_coords = list(range(0, h))
        x_coords = list(range(0, w))

        # Mean value of the pixels along the diagonal
        diagonal_values = [np.mean(img[i, i, :]) for i in range(min(h, w))]

        # Find the first and last points where the threshold is crossed
        first_cross = next(i for i, value in enumerate(diagonal_values) if value >= threshold)
        last_cross = len(diagonal_values) - next(
            i for i, value in enumerate(reversed(diagonal_values)) if value >= threshold)

        # Set the coordinates to crop the image
        y1 = max(0, first_cross)
        y2 = min(h, last_cross)
        x1 = max(0, first_cross)
        x2 = min(w, last_cross)

        # Crop the image using the calculated coordinates
        img_new = cv2.resize(img[y1:y2, x1:x2, :],self.target_size[:2])

        if img_new.shape[0] == 0 or img_new.shape[1] == 0:
            img_new = img
        return img_new

    def extractROI_batch(self,batch_img):
        roi_img_batch=[]
        for img in batch_img:
            roi_img_batch.append(self.extractROI(img))
        return np.asarray(roi_img_batch)

###################################################
######## NOT USED FOR THE FINAL APPROACH ##########
###################################################
        
    def zscore(self,img):
        # Training values
        #Mean pixel value: 140.02717311204663
        #Standard deviation: 60.87266364240796
        mean=140.027
        std=60.8726
        # Apply the z-score formula
        z_scores = (img - mean) / std
        return z_scores
    
    def normalizeIntensities(self,batch_imgs):
        batch_normalized_img=[]
        for img in batch_imgs:
            batch_normalized_img.append(self.zscore(img))
        return np.asarray(batch_normalized_img)

    def segmentationMole(self, batch_imgs):
        batch_imgs_normalized = batch_imgs / 255.0 # Normalize the entire batch
        predicted_masks = self.modelSegmentation.predict(batch_imgs_normalized, verbose=0)# Predict the segmentation mask for the entire batch
        predicted_masks_new=[]
        for img in predicted_masks:
            predicted_masks_new.append(cv2.resize(cv2.GaussianBlur(img, (11,11), 0),(380,380)))
        threshold = 0.5
        predicted_masks_new=np.asarray(predicted_masks_new)
        predicted_masks_binary = (predicted_masks_new > threshold).astype(np.uint8)
        img_binary_rgb = np.stack((predicted_masks_binary, predicted_masks_binary, predicted_masks_binary), axis=-1)
        return img_binary_rgb
    
    def segmentationMoleAndContours(self,batch_imgs):
        batch_imgs_normalized = batch_imgs / 255.0
        predicted_masks = self.modelSegmentation.predict(batch_imgs_normalized, verbose=0)
        threshold = 0.5
        predicted_masks_binary = (predicted_masks > threshold).astype(np.uint8)
        batch_imgs_segmented = predicted_masks_binary * batch_imgs_normalized
        # Initialize array for storing the contours and the structuring elements
        batch_imgs_contours = np.zeros_like(batch_imgs_normalized)
        kernel_dilate = np.ones((11, 11), np.uint8)
        kernel_erode = np.ones((5, 5), np.uint8)
        for i in range(len(predicted_masks_binary)):
            dilated = cv2.dilate(predicted_masks_binary[i], kernel_dilate, iterations=1)
            eroded = cv2.erode(predicted_masks_binary[i], kernel_erode, iterations=1)
            contours = dilated - eroded
            contours_expanded = np.expand_dims(contours, axis=-1)
            batch_imgs_contours[i] = contours_expanded * batch_imgs_normalized[i]
        return batch_imgs_segmented, batch_imgs_contours

    def segment_batch(self,batch_images_resized):
        # Segment the entire batch
        mask_batch=self.segmentationMole(np.asarray(batch_images_resized))
        return np.asarray(mask_batch)

    
