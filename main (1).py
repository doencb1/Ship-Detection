import numpy as np
import cv2
from skimage import io, color
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os
from fcm import FuzzyCMeans
from ssfcm import SSFCM
from fcm_improve import FuzzyCMeansImprove

def convert_normalized_box_to_pixel(box, image_width, image_height):
    x_center, y_center, width, height = box
    x_min = int((x_center - width / 2) * image_width)
    y_min = int((y_center - height / 2) * image_height)
    x_max = int((x_center + width / 2) * image_width)
    y_max = int((y_center + height / 2) * image_height)
    return x_min, y_min, x_max, y_max

def create_mask_from_box(image_height, image_width, box):
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    x_min, y_min, x_max, y_max = box
    mask[y_min:y_max, x_min:x_max] = 1
    return mask

def create_labeled_mask_fcm(image_path, box):
    image = Image.open(image_path)
    image_width, image_height = image.size
    pixel_box = convert_normalized_box_to_pixel(box, image_width, image_height)
    x_min, y_min, x_max, y_max = pixel_box

    cropped_image = np.array(image)[y_min:y_max, x_min:x_max]
    cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    fcm = FuzzyCMeans(n_clusters=2, m=2, max_iter=100, tol=1e-4)
    fcm.fit(cropped_image_gray.reshape(-1, 1))

    fcm_improve = FuzzyCMeansImprove(n_clusters=2, m=2, max_iter=100, tol=1e-4)
    fcm_improve.fit(cropped_image_gray.reshape(-1, 1))

    labels = fcm.predict(cropped_image_gray.reshape(-1, 1)).reshape(cropped_image_gray.shape)
    mask = (labels == 1).astype(np.uint8)  # Giả sử nhãn '1' là tàu
    labels_improve = fcm_improve.predict(cropped_image_gray.reshape(-1, 1)).reshape(cropped_image_gray.shape)
    mask_improve = (labels_improve == 1).astype(np.uint8)  # Giả sử nhãn '1' là tàu

    full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    full_mask[y_min:y_max, x_min:x_max] = mask
    full_mask_improve = np.zeros((image_height, image_width), dtype=np.uint8)
    full_mask_improve[y_min:y_max, x_min:x_max] = mask_improve

    return full_mask, full_mask_improve

def segment_image_with_box(image_path, box, n_clusters=2):
    image = io.imread(image_path)
    image_lab = color.rgb2lab(image)
    image_height, image_width = image.shape[:2]

    pixel_box = convert_normalized_box_to_pixel(box, image_width, image_height)

    labeled_mask = create_mask_from_box(image_height, image_width, pixel_box)

    labeled_pixel_index = np.where(labeled_mask.flatten() == 1)[0]

    image_data = image_lab.reshape((-1, 3)).astype(np.float32) / 255.0

    labeled_mask_fcm, labeled_mask_fcm_improve = create_labeled_mask_fcm(image_path, box)

    sscfm = SSFCM(n_clusters=n_clusters, m=2, alpha=0.5, max_iter=100, tol=0.0001, check=0)
    sscfm.fit(image_data, labeled_pixel_index)

    cs3fcm = SSFCM(n_clusters=n_clusters, m=2, alpha=0.5, max_iter=100, tol=0.0001, check=1)
    cs3fcm.fit(image_data, labeled_mask_fcm)

    ts3fcm = SSFCM(n_clusters=n_clusters, m=2, alpha=0.5, max_iter=100, tol=0.0001, check=2)
    ts3fcm.fit(image_data, labeled_mask_fcm_improve)

    segmented_image_ssfcm = np.argmax(sscfm.U, axis=1).reshape(image_lab.shape[:2])
    segmented_image_cs3fcm = np.argmax(cs3fcm.U, axis=1).reshape(image_lab.shape[:2])
    segmented_image_ts3fcm = np.argmax(ts3fcm.U, axis=1).reshape(image_lab.shape[:2])

    return image, labeled_mask, labeled_mask_fcm, labeled_mask_fcm_improve, segmented_image_ssfcm, segmented_image_cs3fcm, segmented_image_ts3fcm

def display_segmented_image(image, labeled_mask, labeled_mask_fcm, labeled_mask_fcm_improve, segmented_image_ssfcm, segmented_image_cs3fcm, segmented_image_ts3fcm):
    contour_image_ssfcm = image.copy()
    contour_image_cs3fcm = image.copy()
    contour_image_ts3fcm = image.copy()

    for label in np.unique(segmented_image_ssfcm):
        mask = (segmented_image_ssfcm == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_image_ssfcm, contours, -1, (255, 0, 0), 1)

    for label in np.unique(segmented_image_cs3fcm):
        mask = (segmented_image_cs3fcm == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_image_cs3fcm, contours, -1, (255, 0, 0), 1)

    for label in np.unique(segmented_image_ts3fcm):
        mask = (segmented_image_ts3fcm == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_image_ts3fcm, contours, -1, (255, 0, 0), 1)

    plt.figure(figsize=(15, 7))

    plt.subplot(3, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(3, 3, 2)
    plt.imshow(labeled_mask, cmap='gray')
    plt.title('Labeled Mask')
    plt.axis('off')

    plt.subplot(3, 3, 4)
    plt.imshow(labeled_mask_fcm, cmap='gray')
    plt.title('Labeled Mask FCM')
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.imshow(labeled_mask_fcm_improve, cmap='gray')
    plt.title('Labeled Mask FCM Improve')
    plt.axis('off')

    plt.subplot(3, 3, 7)
    plt.imshow(contour_image_ssfcm)
    plt.title('SSFCM')
    plt.axis('off')

    plt.subplot(3, 3, 8)
    plt.imshow(contour_image_cs3fcm)
    plt.title('CS3FCM')
    plt.axis('off')

    plt.subplot(3, 3, 9)
    plt.imshow(contour_image_ts3fcm)
    plt.title('TS3FCM')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

csv_path = 'Label_img.csv'
df = pd.read_csv(csv_path)

n_clusters = 2

image_folder = 'ship'
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    
    if image_path.endswith('.jpg') or image_path.endswith('.png'):
        row = df[df['ImageID'] == image_name.split('.')[0]]
        
        if not row.empty:
            box = (row.iloc[0]['x_center'], row.iloc[0]['y_center'], row.iloc[0]['width'], row.iloc[0]['height'])
            if os.path.exists(image_path):
                image, labeled_mask, labeled_mask_fcm, labeled_mask_fcm_improve, segmented_image_ssfcm, segmented_image_cs3fcm, segmented_image_ts3fcm = segment_image_with_box(image_path, box, n_clusters)
                display_segmented_image(image, labeled_mask, labeled_mask_fcm, labeled_mask_fcm_improve, segmented_image_ssfcm, segmented_image_cs3fcm, segmented_image_ts3fcm)
            else:
                print(f"Image {image_name} not found in {image_folder}")
        else:
            print('Row empty.')



