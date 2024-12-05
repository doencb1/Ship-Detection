import os
import sys
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import time
from fcm import FuzzyCMeans
from ssfcm import SSFCM
from fcm_improve import FuzzyCMeansImprove
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from skimage import io, color
from tkinter import messagebox

# Các hàm đã có...

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

    labeled_data = []
    labels = []
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            labeled_data.append([y, x, cropped_image_gray[y - y_min, x - x_min]])
            labels.append(1)

    labeled_data = np.array(labeled_data)
    labels = np.array(labels)
    
    start_time = time.time()
    fcm = FuzzyCMeans(n_clusters=2, m=2, max_iter=100, tol=1e-4)
    fcm.fit(cropped_image_gray.reshape(-1, 1))
    fcm_time = time.time() - start_time

    start_time = time.time()
    fcm_improve = FuzzyCMeansImprove(n_clusters=2, m=2, max_iter=100, tol=1e-4)
    fcm_improve.fit(cropped_image_gray.reshape(-1, 1))
    fcm_improve_time = time.time()-start_time
    labels = fcm.predict(cropped_image_gray.reshape(-1, 1)).reshape(cropped_image_gray.shape)
    mask = (labels == 1).astype(np.uint8)  # Giả sử nhãn '1' là tàu
    labels_improve = fcm_improve.predict(cropped_image_gray.reshape(-1, 1)).reshape(cropped_image_gray.shape)
    mask_improve = (labels_improve == 1).astype(np.uint8)  # Giả sử nhãn '1' là tàu

    full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    full_mask[y_min:y_max, x_min:x_max] = mask
    full_mask_improve = np.zeros((image_height, image_width), dtype=np.uint8)
    full_mask_improve[y_min:y_max, x_min:x_max] = mask_improve

    times ={
        'fcm_time': fcm_time,
        'fcm_improve_time': fcm_improve_time
    }
    return full_mask, full_mask_improve,times

def segment_image_with_box(image_path, box, n_clusters=2):
    image = io.imread(image_path)
    image_lab = color.rgb2lab(image)
    image_height, image_width = image.shape[:2]

    pixel_box = convert_normalized_box_to_pixel(box, image_width, image_height)

    labeled_mask = create_mask_from_box(image_height, image_width, pixel_box)

    labeled_pixel_index = np.where(labeled_mask.flatten() == 1)[0]

    image_data = image_lab.reshape((-1, 3)).astype(np.float32) / 255.0

    labeled_mask_fcm, labeled_mask_fcm_improve,times = create_labeled_mask_fcm(image_path, box)

    start_time = time.time()
    sscfm = SSFCM(n_clusters=n_clusters, m=2, alpha=0.5, max_iter=100, tol=0.0001, check=0)
    sscfm.fit(image_data, labeled_pixel_index)
    sscfm_time = time.time() - start_time
    
    start_time = time.time()
    cs3fcm = SSFCM(n_clusters=n_clusters, m=2, alpha=0.5, max_iter=100, tol=0.0001, check=1)
    cs3fcm.fit(image_data, labeled_mask_fcm) 
    cs3fcm_time = time.time() - start_time + times["fcm_time"]
    
    start_time = time.time()
    ts3fcm = SSFCM(n_clusters=n_clusters, m=2, alpha=0.5, max_iter=100, tol=0.0001, check=2)
    ts3fcm.fit(image_data, labeled_mask_fcm_improve)
    ts3fcm_time =time.time() - start_time + times["fcm_improve_time"]

    times ={
        'sscfm_time': sscfm_time,
        'cs3fcm_time': cs3fcm_time,
        'ts3fcm_time': ts3fcm_time
    }
    segmented_image_ssfcm = np.argmax(sscfm.U, axis=1).reshape(image_lab.shape[:2])
    segmented_image_cs3fcm = np.argmax(cs3fcm.U, axis=1).reshape(image_lab.shape[:2])
    segmented_image_ts3fcm = np.argmax(ts3fcm.U, axis=1).reshape(image_lab.shape[:2])

    return image, labeled_mask_fcm, labeled_mask_fcm_improve, segmented_image_ssfcm, segmented_image_cs3fcm, segmented_image_ts3fcm,times

def display_segmented_image(image, labeled_mask_fcm, labeled_mask_fcm_improve, segmented_image_ssfcm, segmented_image_cs3fcm, segmented_image_ts3fcm,times):
    
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


    ax3 = fig.add_subplot(3, 3, 4)
    ax4 = fig.add_subplot(3, 3, 5)
    ax5 = fig.add_subplot(3, 3, 7)
    ax6 = fig.add_subplot(3, 3, 8)
    ax7 = fig.add_subplot(3, 3, 9)

    # Hiển thị hình ảnh và tiêu đề tương ứng cho từng trục
    ax3.imshow(labeled_mask_fcm, cmap='gray')
    ax3.set_title('Labeled Mask FCM')
    ax3.axis('off')

    ax4.imshow(labeled_mask_fcm_improve, cmap='gray')
    ax4.set_title('Labeled Mask FCM Improve')
    ax4.axis('off')

    ax5.imshow(contour_image_ssfcm)
    ax5.set_title(f'SSFCM (Time: {times["sscfm_time"]:.2f}s)')
    ax5.axis('off')

    ax6.imshow(contour_image_cs3fcm)
    ax6.set_title(f'CS3FCM (Time: {times["cs3fcm_time"]:.2f}s)')
    ax6.axis('off')

    ax7.imshow(contour_image_ts3fcm)
    ax7.set_title(f'TS3FCM (Time:{times["ts3fcm_time"]:.2f}s)')
    ax7.axis('off')

    # Tạo bố cục chặt chẽ
    fig.tight_layout()
    canvas.draw()
def dislay_img_label(image_path,box):
    fig.clear()
    image = io.imread(image_path)
    image_height, image_width = image.shape[:2]

    pixel_box = convert_normalized_box_to_pixel(box, image_width, image_height)

    labeled_mask = create_mask_from_box(image_height, image_width, pixel_box)
    ax1 = fig.add_subplot(3, 3, 1)
    ax2 = fig.add_subplot(3, 3, 2)
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    ax2.imshow(labeled_mask, cmap='gray')
    ax2.set_title('Labeled Mask')
    ax2.axis('off')
    canvas.draw()

# Chuyển hướng stdout và stderr vào Text widget
class ConsoleOutput:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)

    def flush(self):
        pass  # Không làm gì cả, chỉ để giữ tính tương thích

# Tkinter GUI
root = tk.Tk()
root.title("Image Segmentation")

# Tạo frame chính cho giao diện
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Tạo frame bên trái cho hình ảnh phân đoạn
left_frame = tk.Frame(main_frame, width=600, height=500)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

fig = Figure(figsize=(15, 7))
canvas = FigureCanvasTkAgg(fig, master=left_frame)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Tạo frame bên phải cho các thành phần giao diện khác
right_frame = tk.Frame(main_frame, width=300, height=500)
right_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False)

# Tạo frame con bên trong right_frame cho các thành phần điều khiển phía trên
control_frame = tk.Frame(right_frame)
control_frame.pack(side=tk.TOP, fill=tk.X, expand=False)

# Thành phần chọn ảnh
label_path = tk.StringVar()
label = tk.Label(control_frame, text="Image Path:")
label.pack(side=tk.TOP)

entry = tk.Entry(control_frame, textvariable=label_path)
entry.pack(side=tk.TOP)

csv_path = 'Label_img.csv'
df = pd.read_csv(csv_path)

file_name_label = tk.Label(control_frame, text="File Name: N/A")
file_name_label.pack(side=tk.TOP)

def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
    label_path.set(file_path)
    file_name = os.path.basename(file_path)
    file_name_label.config(text=f"File Name: {file_name}")
    row = df[df['ImageID'] == file_name.split('.')[0]]
    box = (row.iloc[0]['x_center'], row.iloc[0]['y_center'], row.iloc[0]['width'], row.iloc[0]['height'])
    dislay_img_label(file_path,box)

button = tk.Button(control_frame, text="Select Image", command=select_image)
button.pack(side=tk.TOP, padx=5, pady=5)

# Thành phần nhập số lượng cluster
cluster_label = tk.Label(control_frame, text="Number of Clusters:")
cluster_label.pack(side=tk.TOP)

cluster_entry = tk.Entry(control_frame)
cluster_entry.pack(side=tk.TOP)

def show_error_message(message):
    messagebox.showerror("Error", message)
# Nút thực hiện phân đoạn
def segment_image():
    image_path = label_path.get()
    if not image_path.strip():
        show_error_message("Please choose a picture")
        return
    if not cluster_entry.get().strip():  # Kiểm tra xem cluster_value có rỗng không sau khi loại bỏ khoảng trắng
        show_error_message("Cluster entry is empty.")
        return
    n_clusters = int(cluster_entry.get())
    if n_clusters < 2 :
        show_error_message("Number of clusters must be at least 2.")
        return
    file_name = os.path.basename(image_path)
    file_name_label.config(text=f"File Name: {file_name}")
    box = None
    row = df[df['ImageID'] == file_name.split('.')[0]]
    box = (row.iloc[0]['x_center'], row.iloc[0]['y_center'], row.iloc[0]['width'], row.iloc[0]['height'])
    image,labeled_mask_fcm, labeled_mask_fcm_improve, segmented_image_ssfcm, segmented_image_cs3fcm, segmented_image_ts3fcm,times = segment_image_with_box(image_path, box, n_clusters)
    display_segmented_image(image, labeled_mask_fcm, labeled_mask_fcm_improve, segmented_image_ssfcm, segmented_image_cs3fcm, segmented_image_ts3fcm,times)

button_segment = tk.Button(control_frame, text="Segment", command=segment_image)
button_segment.pack(side=tk.TOP, padx=5, pady=5)

# Tạo Text widget cho console bên dưới
console_frame = tk.Frame(right_frame)
console_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

console = tk.Text(console_frame, wrap='word', height=15)
console.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = tk.Scrollbar(console_frame, command=console.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
console['yscrollcommand'] = scrollbar.set

# Chuyển hướng stdout và stderr
sys.stdout = ConsoleOutput(console)
sys.stderr = ConsoleOutput(console)

root.mainloop()
