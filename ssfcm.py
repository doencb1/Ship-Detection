import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
import cv2

class SSFCM:
    def __init__(self, n_clusters=2, m=2, alpha=0.5, max_iter=100, tol=1e-4, check=0):
        self.n_clusters = n_clusters
        self.m = m
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.check = check
        self.centers = None
        self.U = None
        self.f = None
        self.b = None

    def initialize_centers(self, X):
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        centers = X[random_indices]
        return centers

    def initialize_membership_matrix(self, n_samples):
        U = np.random.dirichlet(np.ones(self.n_clusters), size=n_samples)
        return U

    def update_centers(self, X):
        um = self.U ** self.m
        um_alpha = (self.U - self.U * self.b[:, np.newaxis]) ** self.m
        num = np.dot(um.T, X) + self.alpha * np.dot(um_alpha.T, X)
        den = um.sum(axis=0)[:, None] + self.alpha * um_alpha.sum(axis=0)[:, None]
        centers = num / den
        return centers

    def update_membership_matrix(self, X):
        n_samples = X.shape[0]
        new_U = np.zeros((n_samples, self.n_clusters))
        distances = np.zeros((n_samples, self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.linalg.norm(X - self.centers[i], axis=1)
        for k in range(n_samples):
            for i in range(self.n_clusters):
                numerator = 1 + self.alpha * (1 - self.b[k] * np.sum(self.f[k, :]))
                denominator = np.sum([(distances[k, i] / distances[k, j]) ** (2 / (self.m - 1)) for j in range(self.n_clusters)])
                new_U[k, i] = (numerator / denominator) / (1 + self.alpha) + self.alpha * self.f[k, i] * self.b[k]
        return new_U

    def objective_function(self, X):
        n_samples = X.shape[0]
        um = self.U ** self.m
        d = np.zeros((n_samples, self.n_clusters))
        for i in range(self.n_clusters):
            d[:, i] = np.linalg.norm(X - self.centers[i], axis=1)
        term1 = np.sum(um * (d ** 2))
        um_alpha = (self.U - self.U * self.b[:, np.newaxis]) ** self.m
        term2 = self.alpha * np.sum(um_alpha * (d ** 2))
        return term1 + term2

    def fit(self, X, labeled_pixel_index):
        n_samples = X.shape[0]
        self.U = self.initialize_membership_matrix(n_samples)
        self.centers = self.initialize_centers(X)

        self.f = np.zeros((n_samples, self.n_clusters))
        self.b = np.zeros(n_samples)
        self.f[labeled_pixel_index, 0] = 1
        self.b[labeled_pixel_index] = 1

        for iteration in range(self.max_iter):
            self.centers = self.update_centers(X)
            new_U = self.update_membership_matrix(X)
            
            # Tính toán hàm mục tiêu
            objective_value = self.objective_function(X)
            if self.check == 0:
                print(f'SSFCM: {iteration + 1}, Objective function: {objective_value}')
            elif self.check == 1:
                print(f'CS3FCM: {iteration + 1}, Objective function: {objective_value}')
            elif self.check == 2:
                print(f'TS3FCM: {iteration + 1}, Objective function: {objective_value}')
                
            # Kiểm tra điều kiện hội tụ
            if np.linalg.norm(new_U - self.U) < self.tol:
                print(f'Converged after {iteration + 1} iterations')
                break

            self.U = new_U

    def predict(self, X):
        new_U = self.update_membership_matrix(X)
        return np.argmax(new_U, axis=1)

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
#     image_data = image_lab.reshape((-1, 3)).astype(np.float32)

#     # Tìm chỉ số của pixel chứa tàu đã biết
#     labeled_pixel_index = np.where(labeled_mask.flatten() == 1)[0]

#     # Áp dụng phân cụm SSFCM
#     sscfm = SSFCM(n_clusters=n_clusters, m=2, alpha=0.5, max_iter=100, tol=0.0001)
#     sscfm.fit(image_data, labeled_pixel_index)

#     # Phân đoạn ảnh dựa trên trung tâm của các cụm
#     segmented_image = np.argmax(sscfm.U, axis=1).reshape(image_lab.shape[:2])

#     return image, labeled_mask, segmented_image

# def display_segmented_image(image, labeled_mask, segmented_image):
#     contour_image = image.copy()

#     for label in np.unique(segmented_image):
#             mask = (segmented_image == label).astype(np.uint8)
#             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 1)

#     plt.figure(figsize=(18, 6))

#     plt.subplot(1, 3, 1)
#     plt.imshow(image)
#     plt.title('Original Image')
#     plt.axis('off')

#     plt.subplot(1, 3, 2)
#     plt.imshow(labeled_mask, cmap='gray')
#     plt.title('Labeled Mask')
#     plt.axis('off')

#     plt.subplot(1, 3, 3)
#     plt.imshow(contour_image)
#     plt.title('Segmented Image')
#     plt.axis('off')

#     plt.show()

# # Đường dẫn của ảnh và tọa độ box (dạng tỷ lệ)
# image_path = '0b1b3d75b.jpg'
# box = (0.407960, 0.813433, 0.059701, 0.124378)  # (x_center, y_center, width, height)

# # Số lượng cụm mong muốn
# n_clusters = 2

# # Phân đoạn ảnh
# image, labeled_mask, segmented_image = segment_image_with_box(image_path, box, n_clusters)

# # Hiển thị ảnh gốc và ảnh phân đoạn
# display_segmented_image(image, labeled_mask, segmented_image)