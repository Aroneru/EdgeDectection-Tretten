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
            # Konversi BGR (OpenCV) ke RGB (untuk tampilan yang benar di GUI)
            self.rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
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
        return image.copy()

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
            return cv2.Canny(image, 100, 200)
        
        return image

    def calculate_metrics(self, ground_truth, result):
        mse = np.mean((ground_truth - result) ** 2)
        if mse == 0:
            return 0, float('inf')
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        return mse, psnr

# --- KOMPONEN GUI BARU (Zoomable Image) ---
class ZoomableImageFrame(tk.Frame):
    def __init__(self, master=None, title="Image", **kwargs):
        super().__init__(master, **kwargs)
        self.config(bg="white", bd=2, relief="groove")
        
        # Header Frame (Judul + Info Zoom)
        header_frame = tk.Frame(self, bg="white")
        header_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.lbl_title = tk.Label(header_frame, text=title, bg="white", font=("Arial", 10, "bold"))
        self.lbl_title.pack(side=tk.LEFT, padx=5)
        
        self.lbl_zoom = tk.Label(header_frame, text="Zoom: 100%", bg="white", fg="blue", font=("Arial", 9))
        self.lbl_zoom.pack(side=tk.RIGHT, padx=5)

        # Container Canvas
        self.canvas_frame = tk.Frame(self, bg="white")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.vbar = tk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        self.hbar = tk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#ddd", highlightthickness=0,
                                xscrollcommand=self.hbar.set, 
                                yscrollcommand=self.vbar.set)

        self.vbar.config(command=self.canvas.yview)
        self.hbar.config(command=self.canvas.xview)

        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Event Binding untuk Zoom (Ctrl + MouseWheel)
        # Windows
        self.canvas.bind("<Control-MouseWheel>", self.on_zoom)
        # Linux (Button-4 = scroll up, Button-5 = scroll down)
        self.canvas.bind("<Control-Button-4>", self.on_zoom)
        self.canvas.bind("<Control-Button-5>", self.on_zoom)

        self.pil_image = None  # Gambar asli (PIL format)
        self.tk_img = None     # Gambar yang ditampilkan (Tkinter format)
        self.scale = 1.0       # Skala zoom saat ini

    def show_image(self, cv_img):
        self.canvas.delete("all")
        if cv_img is None:
            self.pil_image = None
            self.lbl_zoom.config(text="")
            return

        # Konversi CV image ke PIL image
        # Pastikan mode warna benar (RGB atau L untuk Grayscale)
        if len(cv_img.shape) == 2: # Grayscale
            self.pil_image = Image.fromarray(cv_img)
        else: # RGB
            self.pil_image = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        
        # Reset scale saat gambar baru dimuat
        self.scale = 1.0
        self.redraw_image()

    def on_zoom(self, event):
        if self.pil_image is None:
            return

        # Logika deteksi scroll (Windows & Linux)
        if event.num == 4 or event.delta > 0:
            # Zoom In (Perbesar)
            self.scale *= 1.1
        elif event.num == 5 or event.delta < 0:
            # Zoom Out (Perkecil)
            self.scale /= 1.1

        # --- BATASAN UKURAN ---
        # Batas Maksimum: 100% (Ukuran Asli)
        if self.scale > 2.0:
            self.scale = 2.0
        
        # Batas Minimum: Agar tidak terlalu kecil (misal 5%)
        if self.scale < 0.05:
            self.scale = 0.05

        self.redraw_image()

    def redraw_image(self):
        if self.pil_image is None:
            return

        # Hitung ukuran baru berdasarkan scale
        w, h = self.pil_image.size
        new_w = int(w * self.scale)
        new_h = int(h * self.scale)

        # Resize gambar menggunakan PIL (Resampling.LANCZOS untuk kualitas bagus)
        resized_pil = self.pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized_pil)

        # Gambar ulang di canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_img, anchor='nw')
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
        # Update Label Zoom
        self.lbl_zoom.config(text=f"Zoom: {int(self.scale * 100)}%")

# --- FRONTEND LOGIC (Tampilan Utama) ---
class EdgeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisis Deteksi Tepi (Zoomable)")
        self.root.geometry("1300x800")
        self.processor = ImageProcessor()

        # --- Layout Utama ---
        left_frame = tk.Frame(root, width=250, bg="#f0f0f0", padx=10, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        right_frame = tk.Frame(root, bg="white")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Grid config untuk right frame
        right_frame.columnconfigure(0, weight=1)
        right_frame.columnconfigure(1, weight=1)
        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)

        # --- Kontrol (Kiri) ---
        tk.Label(left_frame, text="Kontrol Panel", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=10)

        btn_load = tk.Button(left_frame, text="Buka Gambar", command=self.load_image, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        btn_load.pack(fill=tk.X, pady=5)

        tk.Label(left_frame, text="Jenis Noise:", bg="#f0f0f0").pack(anchor="w", pady=(15, 0))
        self.noise_type = ttk.Combobox(left_frame, values=["Tidak Ada", "Salt & Pepper", "Gaussian"], state="readonly")
        self.noise_type.current(0)
        self.noise_type.pack(fill=tk.X)

        tk.Label(left_frame, text="Level Noise (0.01 - 0.2):", bg="#f0f0f0").pack(anchor="w", pady=(5, 0))
        self.noise_level = tk.Scale(left_frame, from_=0.0, to=0.2, resolution=0.01, orient=tk.HORIZONTAL, bg="#f0f0f0")
        self.noise_level.set(0.05)
        self.noise_level.pack(fill=tk.X)

        tk.Label(left_frame, text="Operator Deteksi Tepi:", bg="#f0f0f0").pack(anchor="w", pady=(15, 0))
        self.operator_type = ttk.Combobox(left_frame, values=["Canny", "Sobel", "Prewitt", "LoG (Laplace)"], state="readonly")
        self.operator_type.current(0)
        self.operator_type.pack(fill=tk.X)

        btn_process = tk.Button(left_frame, text="Proses Analisis", command=self.process_image, bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
        btn_process.pack(fill=tk.X, pady=20)

        self.lbl_metrics = tk.Label(left_frame, text="Hasil Evaluasi:\nMSE: -\nPSNR: -", justify=tk.LEFT, bg="white", relief="sunken", padx=5, pady=5)
        self.lbl_metrics.pack(fill=tk.X, pady=10)
        
        tk.Label(left_frame, text="Info:\nTahan tombol 'Ctrl'\nsambil scroll mouse\nuntuk Zoom In/Out.", bg="#f0f0f0", fg="gray", font=("Arial", 8)).pack(side=tk.BOTTOM, pady=10)

        # --- Display Gambar (Menggunakan Class Zoomable Baru) ---
        self.panel_original = ZoomableImageFrame(right_frame, title="Citra Asli (Grayscale)")
        self.panel_original.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.panel_noisy = ZoomableImageFrame(right_frame, title="Citra + Noise")
        self.panel_noisy.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.panel_result = ZoomableImageFrame(right_frame, title="Hasil Deteksi Tepi")
        self.panel_result.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
        if file_path:
            if self.processor.load_image(file_path):
                self.panel_original.show_image(self.processor.gray_image)
                self.panel_noisy.show_image(None)
                self.panel_result.show_image(None)
            else:
                messagebox.showerror("Error", "Gagal memuat gambar.")

    def process_image(self):
        if self.processor.gray_image is None:
            messagebox.showwarning("Peringatan", "Harap buka gambar terlebih dahulu.")
            return

        try:
            noise_type = self.noise_type.get()
            noise_amt = self.noise_level.get()
            op_type = self.operator_type.get()

            ground_truth = self.processor.apply_operator(self.processor.gray_image, op_type)

            if noise_type != "Tidak Ada":
                noisy_img = self.processor.add_noise(self.processor.gray_image, noise_type, noise_amt)
            else:
                noisy_img = self.processor.gray_image.copy()
            
            self.panel_noisy.show_image(noisy_img)

            result_edge = self.processor.apply_operator(noisy_img, op_type)
            self.panel_result.show_image(result_edge)

            mse, psnr = self.processor.calculate_metrics(ground_truth, result_edge)
            
            psnr_text = "Inf" if psnr == float('inf') else f"{psnr:.2f} dB"
            self.lbl_metrics.config(text=f"Hasil Evaluasi ({op_type}):\nMSE: {mse:.2f}\nPSNR: {psnr_text}")

        except Exception as e:
            messagebox.showerror("Error", f"Terjadi kesalahan: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = EdgeDetectionApp(root)
    root.mainloop()