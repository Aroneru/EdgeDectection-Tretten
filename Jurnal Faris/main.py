# =========================
# Import library
# =========================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.util import random_noise
import os

# =========================
# List gambar
# =========================
image_folder = "gambar"
image_files = [
    "kupu-geometris.jpg",
    "lidahbuaya-rumit.jpg",
    "tang-gelap.jpg",
    "telur-lengkung.jpg",
    "ijuk-thinline.jpg"
]

# =========================
# Preprocessing
# =========================
def preprocess_image(img_path, noise_type=None, noise_amount=0.01):
    # baca gambar
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Gambar {img_path} tidak ditemukan!")
    
    # ubah ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # resize
    gray = cv2.resize(gray, (256, 256))
    
    # tambahkan noise jika ada
    if noise_type == 'gaussian':
        gray = random_noise(gray, mode='gaussian', var=noise_amount**2)
        gray = (gray * 255).astype(np.uint8)
    elif noise_type == 's&p':
        gray = random_noise(gray, mode='s&p', amount=noise_amount)
        gray = (gray * 255).astype(np.uint8)
    
    return gray

# =========================
# Operator edge detection
# =========================
def sobel_edge(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sobel = cv2.magnitude(sobelx, sobely)
    return sobel.astype(np.uint8)

def prewitt_edge(gray):
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    prewittx = cv2.filter2D(gray, -1, kernelx)
    prewitty = cv2.filter2D(gray, -1, kernely)
    prewitt = cv2.magnitude(prewittx.astype(float), prewitty.astype(float))
    return prewitt.astype(np.uint8)

def log_edge(gray):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    log = cv2.Laplacian(blur, cv2.CV_64F)
    return np.uint8(np.absolute(log))

def canny_edge(gray, low=100, high=200):
    canny = cv2.Canny(gray, low, high)
    return canny

# =========================
# Thresholding Otsu
# =========================
def otsu_threshold(edge_img):
    th = threshold_otsu(edge_img)
    bin_img = (edge_img > th).astype(np.uint8) * 255
    return bin_img

# =========================
# Loop semua gambar + noise + operator
# =========================
for file in image_files:
    path = os.path.join(image_folder, file)
    
    # coba 2 level noise
    for noise_amount in [0.01, 0.05]:
        gray = preprocess_image(path, noise_type='s&p', noise_amount=noise_amount)
        
        # operator
        sobel = sobel_edge(gray)
        prewitt = prewitt_edge(gray)
        log = log_edge(gray)
        canny = canny_edge(gray)
        
        # thresholding Otsu untuk Sobel, Prewitt, LoG
        sobel_bin = otsu_threshold(sobel)
        prewitt_bin = otsu_threshold(prewitt)
        log_bin = otsu_threshold(log)
        
        # tampilkan hasil
        plt.figure(figsize=(12, 6))
        plt.suptitle(f"{file} - Noise: {noise_amount}")
        
        plt.subplot(2, 3, 1)
        plt.imshow(gray, cmap='gray')
        plt.title('Grayscale')
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(sobel_bin, cmap='gray')
        plt.title('Sobel + Otsu')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(prewitt_bin, cmap='gray')
        plt.title('Prewitt + Otsu')
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.imshow(log_bin, cmap='gray')
        plt.title('LoG + Otsu')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(canny, cmap='gray')
        plt.title('Canny')
        plt.axis('off')
        
        plt.show()
