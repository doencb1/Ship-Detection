import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, color
import os
import cv2

class FuzzyCMeansImprove:
    def __init__(self, n_clusters=2, m=2, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters  # Số lượng cụm
        self.m = m  # Tham số mờ m
        self.max_iter = max_iter  # Số lần lặp tối đa
        self.tol = tol  # Ngưỡng hội tụ
        self.centers = None  # Tâm cụm
        self.U = None  # Ma trận độ thuộc
    
    def initialize_centers(self, X):
        """
        Bước 2: Khởi tạo ngẫu nhiên các tâm cụm ban đầu.

        Parameters:
        - X: Ma trận dữ liệu, mỗi hàng là một mẫu.

        Returns:
        - centers: Mảng chứa các tâm cụm được khởi tạo ngẫu nhiên.
        """
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        centers = X[random_indices]
        return centers
    
    def calculate_distance_matrix(self, X, centers):
        """
        Bước 4: Tính ma trận khoảng cách giữa các mẫu dữ liệu và các tâm cụm.

        Parameters:
        - X: Ma trận dữ liệu, mỗi hàng là một mẫu.
        - centers: Mảng chứa các tâm cụm.

        Returns:
        - distance_matrix: Ma trận khoảng cách.
        """
        distance_matrix = np.zeros((X.shape[0], self.n_clusters))
        
        for j in range(self.n_clusters):
            distances = np.linalg.norm(X - centers[j], axis=1)
            distance_matrix[:, j] = distances
        
        return distance_matrix
    
    def update_membership_matrix(self, X, distance_matrix):
        """
        Bước 5: Cập nhật ma trận độ thuộc dựa trên khoảng cách từ mẫu đến các tâm cụm.

        Parameters:
        - X: Ma trận dữ liệu, mỗi hàng là một mẫu.
        - distance_matrix: Ma trận khoảng cách.

        Returns:
        - new_U: Ma trận độ thuộc sau khi cập nhật.
        """
        new_U = np.zeros((X.shape[0], self.n_clusters))
        
        for i in range(X.shape[0]):
            for j in range(self.n_clusters):
                if np.any(distance_matrix[i, :] == 0):
                    # Xử lý trường hợp chia cho 0 bằng cách gán 0 cho new_U[i, j]
                    new_U[i, j] = 0
                else:
                    denominator = np.sum((distance_matrix[i, j] / distance_matrix[i, :]) ** (2 / (self.m - 1)))
                    if np.isnan(denominator):
                        # Xử lý trường hợp giá trị không hợp lệ (NaN) bằng cách gán 0 cho new_U[i, j]
                        new_U[i, j] = 0
                    else:
                        new_U[i, j] = 1 / denominator
        
        return new_U

    def update_membership_strength(self, U):
        """
        Bước 6: Giảm độ thuộc uij^(t) theo công thức.

        Parameters:
        - U: Ma trận độ thuộc hiện tại.

        Returns:
        - new_U: Ma trận độ thuộc sau khi giảm.
        """
        new_U = np.copy(U)
        
        for i in range(U.shape[0]):
            for j in range(self.n_clusters):
                new_U[i, j] /= 2
        
        return new_U

    
    def update_centers(self, X, U):
        """
        Bước 7: Cập nhật các tâm cụm dựa trên độ thuộc và dữ liệu.

        Parameters:
        - X: Ma trận dữ liệu, mỗi hàng là một mẫu.
        - U: Ma trận độ thuộc hiện tại.

        Returns:
        - centers: Mảng chứa các tâm cụm sau khi cập nhật.
        """
        numerator = np.dot(U.T ** self.m, X)
        denominator = np.sum(U ** self.m, axis=0).reshape(-1, 1)
        centers = numerator / denominator
        
        return centers
    
    def check_convergence(self, centers, old_centers):
        """
        Bước 8: Kiểm tra điều kiện hội tụ của thuật toán.

        Parameters:
        - centers: Mảng chứa các tâm cụm hiện tại.
        - old_centers: Mảng chứa các tâm cụm từ lần lặp trước đó.

        Returns:
        - converged: True nếu hội tụ, False nếu chưa hội tụ.
        """
        if np.linalg.norm(centers - old_centers) <= self.tol:
            return True
        else:
            return False
    
    def objective_function(self, X):
        um = self.U ** self.m
        obj = np.sum(um * np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2) ** 2)
        return obj

    def fit(self, X):
        """
        Huấn luyện mô hình FCM trên dữ liệu đã cho.

        Parameters:
        - X: Ma trận dữ liệu, mỗi hàng là một mẫu.
        """
        self.centers = self.initialize_centers(X)
        t = 0  # Bước 1: Khởi tạo biến đếm lần lặp
        
        while t < self.max_iter:
            old_centers = np.copy(self.centers)
            
            # Bước 4: Tính ma trận khoảng cách
            distance_matrix = self.calculate_distance_matrix(X, self.centers)
            
            # Bước 5: Cập nhật ma trận độ thuộc
            self.U = self.update_membership_matrix(X, distance_matrix)
            
            # Bước 6: Giảm độ thuộc
            self.U = self.update_membership_strength(self.U)
            
            # Bước 7: Cập nhật các tâm cụm
            self.centers = self.update_centers(X, self.U)
            
            # Tính giá trị hàm mục tiêu
            objective_value = self.objective_function(X)
            print(f'FCM Improve: {t + 1}, Objective function: {objective_value}')
            
            # Kiểm tra điều kiện hội tụ
            if self.check_convergence(self.centers, old_centers):
                print(f'Converged after {t + 1} iterations.')
                break
            
            # Cập nhật ma trận độ thuộc mới
            new_U = self.update_membership_matrix(X, self.calculate_distance_matrix(X, self.centers))
            if np.linalg.norm(new_U - self.U) < self.tol:
                print(f'Converged after {t + 1} iterations')
                break
            
            self.U = new_U

            t += 1

    def predict(self, X):
        """
        Dự đoán nhãn của các mẫu dữ liệu mới.

        Parameters:
        - X: Ma trận dữ liệu mới cần dự đoán.

        Returns:
        - predicted_labels: Mảng chứa nhãn dự đoán cho từng mẫu dữ liệu.
        """
        distance_matrix = self.calculate_distance_matrix(X, self.centers)
        new_U = self.update_membership_matrix(X, distance_matrix)
        predicted_labels = np.argmax(new_U, axis=1)
        
        return predicted_labels

# def convert_normalized_box_to_pixel(box, image_width, image_height):
#     x_center, y_center, width, height = box
#     x_min = int((x_center - width / 2) * image_width)
#     y_min = int((y_center - height / 2) * image_height)
#     x_max = int((x_center + width / 2) * image_width)
#     y_max = int((y_center + height / 2) * image_height)
#     return x_min, y_min, x_max, y_max

# def create_mask_from_box(image_height, image_width, box):
#     mask = np.zeros((image_height, image_width), dtype=np.uint8)
#     x_min, y_min, x_max, y_max = box
#     mask[y_min:y_max, x_min:x_max] = 1
#     return mask

# def segment_image_with_box(image_path, box, n_clusters=2):
#     # Đọc ảnh đầu vào
#     image = io.imread(image_path)
#     image_lab = color.rgb2lab(image)
#     image_height, image_width = image.shape[:2]

#     # Chuyển đổi tọa độ nhãn từ dạng tỷ lệ sang pixel
#     pixel_box = convert_normalized_box_to_pixel(box, image_width, image_height)

#     # Tạo mask từ box
#     labeled_mask = create_mask_from_box(image_height, image_width, pixel_box)

#     # Chuẩn bị dữ liệu đầu vào cho phân cụm bằng cách chuyển đổi ảnh sang không gian màu LAB và chuẩn hóa các giá trị
#     image_data = image_lab.reshape((-1, 3)).astype(np.float32) / 255.0

#     fcm = ImprovedFCM(n_clusters=n_clusters, m=2, max_iter=100, tol=0.0001)
#     fcm.fit(image_data)

#     # Phân đoạn ảnh dựa trên trung tâm của các cụm
#     segmented_image = np.argmax(fcm.U, axis=1).reshape(image_lab.shape[:2])

#     return image, labeled_mask, segmented_image

# def display_segmented_image(image, labeled_mask, segmented_image_fcm):
#     contour_image_fcm = image.copy()

#     for label in np.unique(segmented_image_fcm):
#         mask = (segmented_image_fcm == label).astype(np.uint8)
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(contour_image_fcm, contours, -1, (255, 0, 0), 1)
        
#     plt.figure(figsize=(18, 7))

#     plt.subplot(2, 3, 1)
#     plt.imshow(image)
#     plt.title('Original Image')
#     plt.axis('off')

#     plt.subplot(2, 3, 2)
#     plt.imshow(labeled_mask, cmap='gray')
#     plt.title('Labeled Mask')
#     plt.axis('off')

#     plt.subplot(2, 3, 3)
#     plt.imshow(contour_image_fcm)
#     plt.title('Segmented Image (Improved FCM)')
#     plt.axis('off')

#     plt.tight_layout()
#     plt.show()

# # Đường dẫn của ảnh và tọa độ box (dạng tỷ lệ)
# image_path = 'D:/TinhToanMem/Nhom8/Code/ship/d6b2845e7.jpg'
# box = (0.407960, 0.813433, 0.059701, 0.124378)  # (x_center, y_center, width, height)

# # Số lượng cụm mong muốn
# n_clusters = 2

# # Phân đoạn ảnh
# image, labeled_mask, segmented_image = segment_image_with_box(image_path, box, n_clusters)

# # Hiển thị ảnh gốc và ảnh phân đoạn
# display_segmented_image(image, labeled_mask, segmented_image)    

# -----------------------------------
# # Sử dụng để phân đoạn từng ảnh từ tập dữ liệu
# csv_path = 'Label_img.csv'
# df = pd.read_csv(csv_path)

# n_clusters = 2  # Số lượng cụm mong muốn

# for _, row in df.iterrows():
#     image_name = row['ImageID'] + ".jpg"
#     box = (row['x_center'], row['y_center'], row['width'], row['height'])

#     image_path = os.path.join('selected_images', image_name)
    
#     if os.path.exists(image_path):
#         image, labeled_mask, segmented_image = segment_image_with_box(image_path, box, n_clusters)
#         display_segmented_image(image, labeled_mask, segmented_image)
#     else:
#         print(f"Image {image_name} not found in {'selected_images'}")

# # Hoặc sử dụng cho từng ảnh trong thư mục
# image_folder = 'ship'
# for image_name in os.listdir(image_folder):
#     image_path = os.path.join(image_folder, image_name)
    
#     if image_path.endswith('.jpg') or image_path.endswith('.png'):
#         # Tìm thông tin box trong CSV
#         row = df[df['ImageID'] == image_name.split('.')[0]]
        
#         if not row.empty:
#             box = (row.iloc[0]['x_center'], row.iloc[0]['y_center'], row.iloc[0]['width'], row.iloc[0]['height'])
#         else:
#             box = None

#         if os.path.exists(image_path):
#             image, labeled_mask, segmented_image_fcm = segment_image_with_box(image_path, box, n_clusters)
#             display_segmented_image(image, labeled_mask, segmented_image_fcm)
#         else:
#             print(f"Image {image_name} not found in {image_folder}")

