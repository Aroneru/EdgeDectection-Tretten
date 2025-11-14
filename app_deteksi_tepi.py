import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import math
from skimage.util import random_noise

# --- BACKEND LOGIC (Pemrosesan Citra) ---
class ImageProcessor:
    def __init__(self):
        self.original_image = None
        self.gray_image = None

    def load_image(self, path):
        self.original_image = cv2.imread(path)
        if self.original_image is not None:
            self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            return True
        return False

    def add_noise(self, image, noise_type, amount):
        if noise_type == "Salt & Pepper":
            noisy = random_noise(image, mode='s&p', amount=amount)
            return (255 * noisy).astype(np.uint8)
        elif noise_type == "Gaussian":
            noisy = random_noise(image, mode='gaussian', var=amount)
            return (255 * noisy).astype(np.uint8)
        return image.copy() # Tidak ada noise

    def get_otsu_threshold(self, image):
        val, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def apply_operator(self, image, operator_name):
        if operator_name == "Sobel":
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            grad = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, 
                                   cv2.convertScaleAbs(grad_y), 0.5, 0)
            return self.get_otsu_threshold(grad)

        elif operator_name == "Prewitt":
            kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            img_prewittx = cv2.filter2D(image, -1, kernelx)
            img_prewitty = cv2.filter2D(image, -1, kernely)
            grad = cv2.add(img_prewittx, img_prewitty)
            return self.get_otsu_threshold(grad)

        elif operator_name == "LoG (Laplace)":
            blurred = cv2.GaussianBlur(image, (3, 3), 0)
            laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
            return self.get_otsu_threshold(cv2.convertScaleAbs(laplacian))

        elif operator_name == "Canny":
            # Canny menggunakan Hysteresis thresholding bawaan jadi gak ditamabahin otsu kalau kamu nanya
            return cv2.Canny(image, 100, 200)
        
        return image

    def calculate_metrics(self, ground_truth, result):
        mse = np.mean((ground_truth - result) ** 2)
        if mse == 0:
            return 0, float('inf')
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        return mse, psnr

# --- FRONTEND LOGIC (Tampilan GUI) ---
class EdgeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisis Deteksi Tepi & Threshold Optimal")
        self.root.geometry("1200x700")
        self.processor = ImageProcessor()

        # --- Layout Utama ---
        left_frame = tk.Frame(root, width=300, bg="#f0f0f0", padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        right_frame = tk.Frame(root, bg="white")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Kontrol (Kiri) ---
        tk.Label(left_frame, text="Kontrol Panel", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=10)

        # Tombol Load
        btn_load = tk.Button(left_frame, text="Buka Gambar", command=self.load_image, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        btn_load.pack(fill=tk.X, pady=5)

        # Pengaturan Noise
        tk.Label(left_frame, text="Jenis Noise:", bg="#f0f0f0").pack(anchor="w", pady=(15, 0))
        self.noise_type = ttk.Combobox(left_frame, values=["Tidak Ada", "Salt & Pepper", "Gaussian"], state="readonly")
        self.noise_type.current(0)
        self.noise_type.pack(fill=tk.X)

        tk.Label(left_frame, text="Level Noise (0.01 - 0.2):", bg="#f0f0f0").pack(anchor="w", pady=(5, 0))
        self.noise_level = tk.Scale(left_frame, from_=0.0, to=0.2, resolution=0.01, orient=tk.HORIZONTAL, bg="#f0f0f0")
        self.noise_level.set(0.05)
        self.noise_level.pack(fill=tk.X)

        # Milih Operator
        tk.Label(left_frame, text="Operator Deteksi Tepi:", bg="#f0f0f0").pack(anchor="w", pady=(15, 0))
        self.operator_type = ttk.Combobox(left_frame, values=["Canny", "Sobel", "Prewitt", "LoG (Laplace)"], state="readonly")
        self.operator_type.current(0)
        self.operator_type.pack(fill=tk.X)

        # Tombol Proses
        btn_process = tk.Button(left_frame, text="Proses Analisis", command=self.process_image, bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
        btn_process.pack(fill=tk.X, pady=20)

        # Hasil Metrik
        self.lbl_metrics = tk.Label(left_frame, text="Hasil Evaluasi:\nMSE: -\nPSNR: -", justify=tk.LEFT, bg="white", relief="sunken", padx=5, pady=5)
        self.lbl_metrics.pack(fill=tk.X, pady=10)

        # --- Display Gambar (Kanan) --- (still buggy)
        self.panel_original = self.create_image_panel(right_frame, "Citra Asli (Grayscale)")
        self.panel_original.grid(row=0, column=0, padx=10, pady=10)

        self.panel_noisy = self.create_image_panel(right_frame, "Citra + Noise")
        self.panel_noisy.grid(row=0, column=1, padx=10, pady=10)

        self.panel_result = self.create_image_panel(right_frame, "Hasil Deteksi Tepi (Binary)")
        self.panel_result.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

    def create_image_panel(self, parent, title):
        frame = tk.Frame(parent, bg="white")
        lbl_title = tk.Label(frame, text=title, bg="white", font=("Arial", 10, "bold"))
        lbl_title.pack()
        lbl_img = tk.Label(frame, bg="#ddd", text="No Image", width=40, height=15)
        lbl_img.pack()
        return frame

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if file_path:
            if self.processor.load_image(file_path):
                self.show_image(self.processor.gray_image, self.panel_original.winfo_children()[1])
                # Reset display lain
                self.show_image(None, self.panel_noisy.winfo_children()[1])
                self.show_image(None, self.panel_result.winfo_children()[1])
            else:
                messagebox.showerror("Error", "Gagal memuat gambar.")

    def show_image(self, cv_img, label_widget):
        if cv_img is None:
            label_widget.config(image='', text="No Image")
            return

        # Resize agar muat di GUI
        h, w = cv_img.shape
        target_size = (350, 350)
        scale = min(target_size[0]/w, target_size[1]/h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(cv_img, (new_w, new_h))
        img_pil = Image.fromarray(resized)
        img_tk = ImageTk.PhotoImage(img_pil)
        
        label_widget.config(image=img_tk, text="")
        label_widget.image = img_tk # Keep reference

    def process_image(self):
        if self.processor.gray_image is None:
            messagebox.showwarning("Peringatan", "Harap buka gambar terlebih dahulu.")
            return

        try:
            # Get parameter
            noise_type = self.noise_type.get()
            noise_amt = self.noise_level.get()
            op_type = self.operator_type.get()

            # Buat Ground Truth (Deteksi tepi pada citra bersih) buat variabel kontrol
            ground_truth = self.processor.apply_operator(self.processor.gray_image, op_type)

            # Buat Citra Noisy
            if noise_type != "Tidak Ada":
                noisy_img = self.processor.add_noise(self.processor.gray_image, noise_type, noise_amt)
            else:
                noisy_img = self.processor.gray_image.copy()
            
            self.show_image(noisy_img, self.panel_noisy.winfo_children()[1])

            # Terapkan Operator pada Citra Noisy
            result_edge = self.processor.apply_operator(noisy_img, op_type)
            
            self.show_image(result_edge, self.panel_result.winfo_children()[1])

            # Hitung Metrik (MSE & PSNR)
            mse, psnr = self.processor.calculate_metrics(ground_truth, result_edge)
            
            psnr_text = "Inf" if psnr == float('inf') else f"{psnr:.2f} dB"
            self.lbl_metrics.config(text=f"Hasil Evaluasi ({op_type}):\nMSE: {mse:.2f}\nPSNR: {psnr_text}")

        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = EdgeDetectionApp(root)
    root.mainloop()