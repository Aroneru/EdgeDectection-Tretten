import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import math
from skimage.util import random_noise
import pillow_heif  

# Mendaftarkan opener HEIC agar PIL bisa membacanya langsung
pillow_heif.register_heif_opener()

# --- BACKEND LOGIC (Pemrosesan Citra) ---
class ImageProcessor:
    def __init__(self):
        self.original_image = None # BGR Format (OpenCV standard)
        self.rgb_image = None      # RGB Format (Untuk Display GUI)
        self.gray_image = None     # Grayscale Format (Untuk Proses Deteksi)

    def load_image(self, path):
        # 1. Coba load menggunakan OpenCV standar (JPG, PNG, BMP)
        self.original_image = cv2.imread(path)

        # 2. Jika OpenCV gagal (biasanya return None untuk HEIC), gunakan Pillow+Heif
        if self.original_image is None:
            try:
                # Buka menggunakan Pillow (yang sudah support HEIC via register_heif_opener)
                pil_img = Image.open(path)
                
                # Konversi ke Numpy Array (Format RGB)
                rgb_array = np.array(pil_img)
                
                # Konversi RGB ke BGR agar formatnya seragam dengan OpenCV
                self.original_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"Gagal membuka gambar: {e}")
                return False

        # 3. Jika berhasil dimuat (baik via OpenCV atau Pillow)
        if self.original_image is not None:
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

# --- KOMPONEN GUI (Zoomable Image Frame) ---
class ZoomableImageFrame(tk.Frame):
    def __init__(self, master=None, title="Image", **kwargs):
        super().__init__(master, **kwargs)
        self.config(bg="white", bd=2, relief="groove")
        
        # Header
        header_frame = tk.Frame(self, bg="white")
        header_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.lbl_title = tk.Label(header_frame, text=title, bg="white", font=("Arial", 10, "bold"))
        self.lbl_title.pack(side=tk.LEFT, padx=5)
        
        self.lbl_zoom = tk.Label(header_frame, text="Zoom: 100%", bg="white", fg="blue", font=("Arial", 9))
        self.lbl_zoom.pack(side=tk.RIGHT, padx=5)

        # Canvas
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
        
        # Zoom Binding
        self.canvas.bind("<Control-MouseWheel>", self.on_zoom) 
        self.canvas.bind("<Control-Button-4>", self.on_zoom)   
        self.canvas.bind("<Control-Button-5>", self.on_zoom)   

        self.pil_image = None 
        self.tk_img = None     
        self.scale = 1.0       

    def show_image(self, cv_img):
        self.canvas.delete("all")
        if cv_img is None:
            self.pil_image = None
            self.lbl_zoom.config(text="")
            return

        if len(cv_img.shape) == 2: 
            self.pil_image = Image.fromarray(cv_img)
        else: 
            self.pil_image = Image.fromarray(cv_img) 
        
        # Hitung skala agar fit to frame
        self.scale = self.calculate_fit_scale()
        self.redraw_image()

    def calculate_fit_scale(self):
        """Menghitung skala agar gambar fit ke canvas"""
        if self.pil_image is None:
            return 1.0
        
        # Update canvas agar mendapat ukuran terbaru
        self.canvas.update_idletasks()
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Jika canvas belum di-render (width/height = 1), gunakan ukuran default
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 600
            canvas_height = 400
        
        img_width, img_height = self.pil_image.size
        
        # Hitung skala berdasarkan lebar dan tinggi
        scale_width = canvas_width / img_width
        scale_height = canvas_height / img_height
        
        # Pilih skala terkecil agar gambar tidak terpotong
        scale = min(scale_width, scale_height)
        
        # Batasi maksimal skala = 1.0 (tidak memperbesar)
        return min(scale, 1.0)

    def on_zoom(self, event):
        if self.pil_image is None: return
        
        if event.num == 4 or event.delta > 0:
            self.scale *= 1.1
        elif event.num == 5 or event.delta < 0:
            self.scale /= 1.1

        # Hapus batasan scale > 1.0 agar bisa zoom in lebih besar
        # if self.scale > 1.0: self.scale = 1.0
        if self.scale < 0.05: self.scale = 0.05

        self.redraw_image()

    def redraw_image(self):
        if self.pil_image is None: return

        w, h = self.pil_image.size
        new_w = int(w * self.scale)
        new_h = int(h * self.scale)

        resized_pil = self.pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized_pil)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_img, anchor='nw')
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        self.lbl_zoom.config(text=f"Zoom: {int(self.scale * 100)}%")

# --- FRONTEND LOGIC (Main App) ---
class EdgeDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Edge Detection Analysis Tool")
        self.root.geometry("1300x850")
        self.processor = ImageProcessor()

        left_frame = tk.Frame(root, width=250, bg="#f0f0f0", padx=8, pady=10)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)
        left_frame.pack_propagate(False)  # Mencegah frame menyesuaikan ukuran dengan konten
        
        right_frame = tk.Frame(root, bg="white")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        right_frame.columnconfigure(0, weight=1)
        right_frame.columnconfigure(1, weight=1)
        right_frame.rowconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)

        # --- KONTROL PANEL ---
        tk.Label(left_frame, text="Control Panel", font=("Arial", 14, "bold"), bg="#f0f0f0").pack(pady=10)

        btn_load = tk.Button(left_frame, text="Select Image", command=self.load_image, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        btn_load.pack(fill=tk.X, pady=5)

        tk.Label(left_frame, text="Noise Type:", bg="#f0f0f0").pack(anchor="w", pady=(15, 0))
        self.noise_type = ttk.Combobox(left_frame, values=["No Noise", "Salt & Pepper", "Gaussian"], state="readonly")
        self.noise_type.current(0)
        self.noise_type.pack(fill=tk.X)

        tk.Label(left_frame, text="Noise Level (0.01 - 0.2):", bg="#f0f0f0").pack(anchor="w", pady=(5, 0))
        self.noise_level = tk.Scale(left_frame, from_=0.0, to=0.2, resolution=0.01, orient=tk.HORIZONTAL, bg="#f0f0f0")
        self.noise_level.set(0.05)
        self.noise_level.pack(fill=tk.X)

        tk.Label(left_frame, text="Single Operator Selection:", bg="#f0f0f0").pack(anchor="w", pady=(15, 0))
        self.operator_type = ttk.Combobox(left_frame, values=["Canny", "Sobel", "Prewitt", "LoG (Laplace)"], state="readonly")
        self.operator_type.current(0)
        self.operator_type.pack(fill=tk.X)

        # Tombol Proses Tunggal
        btn_process = tk.Button(left_frame, text="Analyze Single Operator", command=self.process_image, bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
        btn_process.pack(fill=tk.X, pady=(10, 5))

        # --- TOMBOL BARU: BANDINGKAN SEMUA ---
        btn_compare_all = tk.Button(left_frame, text="Process & Compare All", command=self.process_all_operators, bg="#FF9800", fg="white", font=("Arial", 10, "bold"))
        btn_compare_all.pack(fill=tk.X, pady=5)

        # Frame untuk Text widget dengan scrollbar
        metrics_frame = tk.Frame(left_frame, bg="white", relief="sunken", bd=2)
        metrics_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(metrics_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Text widget (menggantikan Label)
        self.lbl_metrics = tk.Text(metrics_frame, wrap=tk.WORD, height=12, 
                                   yscrollcommand=scrollbar.set,
                                   font=("Courier", 9), padx=5, pady=5,
                                   bg="white", fg="black")
        self.lbl_metrics.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.lbl_metrics.yview)
        
        # Set initial text
        self.lbl_metrics.insert("1.0", "Evaluation Results:\nMSE: -\nPSNR: -")
        self.lbl_metrics.config(state=tk.DISABLED)  # Read-only
        
        tk.Label(left_frame, text="Info:\n- Support JPG, PNG, HEIC\n- Ctrl + Scroll to Zoom", bg="#f0f0f0", fg="gray", font=("Arial", 8)).pack(side=tk.BOTTOM, pady=10)

        # --- DISPLAY PANEL (Main Window) ---
        self.panel_rgb = ZoomableImageFrame(right_frame, title="1. Original Image (RGB)")
        self.panel_rgb.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.panel_gray = ZoomableImageFrame(right_frame, title="2. Grayscale Image (Input)")
        self.panel_gray.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.panel_noisy = ZoomableImageFrame(right_frame, title="3. Image + Noise")
        self.panel_noisy.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.panel_result = ZoomableImageFrame(right_frame, title="4. Single Edge Result")
        self.panel_result.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        self.comparison_window = None # Store reference to popup

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.heic;*.HEIC")])
        if file_path:
            if self.processor.load_image(file_path):
                self.panel_rgb.show_image(self.processor.rgb_image)
                self.panel_gray.show_image(self.processor.gray_image)
                
                self.panel_noisy.show_image(None)
                self.panel_result.show_image(None)
                self.update_metrics_text("Evaluation Results:\nMSE: -\nPSNR: -")
            else:
                messagebox.showerror("Error", "Failed to load image.")

    def update_metrics_text(self, text):
        """Helper method to update Text widget"""
        self.lbl_metrics.config(state=tk.NORMAL)
        self.lbl_metrics.delete("1.0", tk.END)
        self.lbl_metrics.insert("1.0", text)
        self.lbl_metrics.config(state=tk.DISABLED)

    def process_image(self):
        """Proses hanya 1 operator (sesuai pilihan dropdown)"""
        if self.processor.gray_image is None:
            messagebox.showwarning("Warning", "Please select an image first.")
            return

        try:
            noise_type = self.noise_type.get()
            noise_amt = self.noise_level.get()
            op_type = self.operator_type.get()

            # Ground truth (tanpa noise)
            ground_truth = self.processor.apply_operator(self.processor.gray_image, op_type)

            # Tambahkan noise
            if noise_type != "No Noise":
                noisy_img = self.processor.add_noise(self.processor.gray_image, noise_type, noise_amt)
            else:
                noisy_img = self.processor.gray_image.copy()
            
            self.panel_noisy.show_image(noisy_img)

            # Deteksi tepi pada gambar noisy
            result_edge = self.processor.apply_operator(noisy_img, op_type)
            self.panel_result.show_image(result_edge)

            # Hitung metrik
            mse, psnr = self.processor.calculate_metrics(ground_truth, result_edge)
            
            # Format tampilan
            psnr_text = "Inf" if psnr == float('inf') else f"{psnr:.2f}"
            
            metrics_text = f"Single Result ({op_type}):\n"
            metrics_text += f"{'='*28}\n"
            metrics_text += f"Noise: {noise_type}\n"
            metrics_text += f"Level: {noise_amt}\n"
            metrics_text += f"{'='*28}\n"
            metrics_text += f"MSE: {mse:.4f}\n"
            metrics_text += f"PSNR: {psnr_text} dB"
            
            self.update_metrics_text(metrics_text)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()

    def process_all_operators(self):
        """Proses 4 operator sekaligus dan tampilkan di jendela baru"""
        if self.processor.gray_image is None:
            messagebox.showwarning("Warning", "Please select an image first.")
            return

        # Tutup jendela lama jika ada
        if self.comparison_window and self.comparison_window.winfo_exists():
            self.comparison_window.destroy()

        try:
            noise_type = self.noise_type.get()
            noise_amt = self.noise_level.get()
            operators = ["Canny", "Sobel", "Prewitt", "LoG (Laplace)"]
            
            # 1. Siapkan Gambar Noisy
            if noise_type != "No Noise":
                noisy_img = self.processor.add_noise(self.processor.gray_image, noise_type, noise_amt)
            else:
                noisy_img = self.processor.gray_image.copy()
            
            # Update panel noisy di main window agar user tahu inputnya apa
            self.panel_noisy.show_image(noisy_img)

            # 2. Buat Jendela Pop-up
            self.comparison_window = tk.Toplevel(self.root)
            self.comparison_window.title(f"Comparison: {noise_type} (Level {noise_amt})")
            self.comparison_window.geometry("1200x800")
            
            # Config Grid Pop-up
            self.comparison_window.columnconfigure(0, weight=1)
            self.comparison_window.columnconfigure(1, weight=1)
            self.comparison_window.rowconfigure(0, weight=1)
            self.comparison_window.rowconfigure(1, weight=1)

            # Panel-panel di jendela pop-up
            panels = {}
            positions = [(0,0), (0,1), (1,0), (1,1)] # Grid positions
            
            # Header untuk metrics summary
            metrics_summary = f"COMPARE ALL OPERATORS\n"
            metrics_summary += f"{'='*28}\n"
            metrics_summary += f"Noise Type: {noise_type}\n"
            metrics_summary += f"Noise Level: {noise_amt}\n"
            metrics_summary += f"{'='*28}\n"

            for i, op_name in enumerate(operators):
                # A. Hitung Ground Truth dan Result
                ground_truth = self.processor.apply_operator(self.processor.gray_image, op_name)
                result_edge = self.processor.apply_operator(noisy_img, op_name)
                
                # B. Hitung Metrik
                mse, psnr = self.processor.calculate_metrics(ground_truth, result_edge)
                
                # C. Format Metrik
                psnr_text = "Inf" if psnr == float('inf') else f"{psnr:.2f}"
                metrics_summary += f"\n{op_name}:\n"
                metrics_summary += f"  MSE  : {mse:.4f}\n"
                metrics_summary += f"  PSNR : {psnr_text} dB\n"

                # D. Buat Panel dan Tampilkan
                frame_title = f"{op_name} | MSE: {mse:.2f} | PSNR: {psnr_text} dB"
                panel = ZoomableImageFrame(self.comparison_window, title=frame_title)
                r, c = positions[i]
                panel.grid(row=r, column=c, padx=5, pady=5, sticky="nsew")
                panel.show_image(result_edge)
                panels[op_name] = panel

            # 3. Tampilkan Metrik di Main Window
            self.update_metrics_text(metrics_summary)

        except Exception as e:
            messagebox.showerror("Error", f"Comparison failed: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = EdgeDetectionApp(root)
    root.mainloop()