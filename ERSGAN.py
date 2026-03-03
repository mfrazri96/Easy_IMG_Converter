import cv2
import os
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import torch
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

class ESRGANImageEnhancer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Real-ESRGAN Image Enhancer")
        self.root.geometry("600x500")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.model_name = tk.StringVar(value="RealESRGAN_x4plus")
        self.scale_factor = tk.IntVar(value=4)
        self.progress_var = tk.StringVar(value="Ready")
        
        # Model configurations
        self.model_configs = {
            'RealESRGAN_x4plus': {
                'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
                'scale': 4,
                'model_path': 'weights/RealESRGAN_x4plus.pth',
                'download_url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
            },
            'RealESRGAN_x2plus': {
                'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2),
                'scale': 2,
                'model_path': 'weights/RealESRGAN_x2plus.pth',
                'download_url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
            },
            'RealESRGAN_x4plus_anime_6B': {
                'model': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4),
                'scale': 4,
                'model_path': 'weights/RealESRGAN_x4plus_anime_6B.pth',
                'download_url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth'
            }
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Real-ESRGAN Image Enhancer", 
                               font=('Helvetica', 18, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Input file selection
        input_frame = ttk.LabelFrame(main_frame, text="Input Image", padding="10")
        input_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Entry(input_frame, textvariable=self.input_path, width=60).grid(row=0, column=0, padx=5)
        ttk.Button(input_frame, text="Browse Image", command=self.browse_input).grid(row=0, column=1, padx=5)
        
        # Output file selection
        output_frame = ttk.LabelFrame(main_frame, text="Output Location", padding="10")
        output_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Entry(output_frame, textvariable=self.output_path, width=60).grid(row=0, column=0, padx=5)
        ttk.Button(output_frame, text="Save As", command=self.browse_output).grid(row=0, column=1, padx=5)
        
        # Model selection
        model_frame = ttk.LabelFrame(main_frame, text="Enhancement Settings", padding="10")
        model_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(model_frame, text="Model:").grid(row=0, column=0, sticky=tk.W)
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_name, 
                                  values=list(self.model_configs.keys()), 
                                  state="readonly", width=30)
        model_combo.grid(row=0, column=1, padx=5, sticky=tk.W)
        model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        ttk.Label(model_frame, text="Scale Factor:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        scale_combo = ttk.Combobox(model_frame, textvariable=self.scale_factor,
                                  values=[2, 4], state="readonly", width=10)
        scale_combo.grid(row=1, column=1, padx=5, pady=(10, 0), sticky=tk.W)
        
        # Model info
        self.info_text = tk.Text(model_frame, height=4, width=70, wrap=tk.WORD)
        self.info_text.grid(row=2, column=0, columnspan=3, pady=(10, 0), sticky=(tk.W, tk.E))
        self.update_model_info()
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=20)
        
        ttk.Button(button_frame, text="Check Model", command=self.check_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Enhance Image", command=self.start_enhancement, 
                  style='Accent.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_fields).pack(side=tk.LEFT, padx=5)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        progress_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.status_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        self.status_label.grid(row=1, column=0, sticky=tk.W)
        
        # Log frame
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.log_text = tk.Text(log_frame, height=8, width=70)
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(6, weight=1)
        progress_frame.columnconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
    
    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.input_path.set(filename)
            # Auto-generate output filename
            base_name = os.path.splitext(filename)[0]
            model_name = self.model_name.get()
            scale = self.scale_factor.get()
            output_name = f"{base_name}_enhanced_{model_name}_x{scale}.png"
            self.output_path.set(output_name)
    
    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Save Enhanced Image As",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.output_path.set(filename)
    
    def on_model_change(self, event):
        model_name = self.model_name.get()
        if model_name in self.model_configs:
            self.scale_factor.set(self.model_configs[model_name]['scale'])
        self.update_model_info()
        
        # Update output filename if input is selected
        if self.input_path.get():
            base_name = os.path.splitext(self.input_path.get())[0]
            scale = self.scale_factor.get()
            output_name = f"{base_name}_enhanced_{model_name}_x{scale}.png"
            self.output_path.set(output_name)
    
    def update_model_info(self):
        model_name = self.model_name.get()
        info_text = {
            'RealESRGAN_x4plus': 'Best for: Real photos and general images\nScale: 4x upscaling\nUse for: Photographs, scanned documents, realistic images',
            'RealESRGAN_x2plus': 'Best for: Moderate enhancement\nScale: 2x upscaling\nUse for: When file size matters or moderate enhancement needed',
            'RealESRGAN_x4plus_anime_6B': 'Best for: Anime and cartoon images\nScale: 4x upscaling\nUse for: Anime screenshots, drawn artwork, illustrations'
        }
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text.get(model_name, 'Model information not available'))
        self.info_text.config(state=tk.DISABLED)
    
    def log_message(self, message):
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def check_model(self):
        model_name = self.model_name.get()
        if model_name not in self.model_configs:
            messagebox.showerror("Error", f"Unknown model: {model_name}")
            return
        
        config = self.model_configs[model_name]
        model_path = config['model_path']
        
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            messagebox.showinfo("Model Status", f"✅ Model found!\nPath: {model_path}\nSize: {file_size:.1f} MB")
            self.log_message(f"✅ Model {model_name} is available")
        else:
            download_url = config['download_url']
            result = messagebox.askyesno("Model Not Found", 
                                       f"❌ Model file not found: {model_path}\n\n"
                                       f"Download URL:\n{download_url}\n\n"
                                       f"Would you like to open the download page?")
            if result:
                import webbrowser
                webbrowser.open(download_url)
            
            self.log_message(f"❌ Model {model_name} not found at: {model_path}")
    
    def clear_fields(self):
        self.input_path.set("")
        self.output_path.set("")
        self.log_text.delete(1.0, tk.END)
        self.progress_var.set("Ready")
    
    def start_enhancement(self):
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input image")
            return
        
        if not self.output_path.get():
            messagebox.showerror("Error", "Please specify output location")
            return
        
        # Start enhancement in a separate thread to prevent GUI freezing
        thread = threading.Thread(target=self.enhance_image_thread)
        thread.daemon = True
        thread.start()
    
    def enhance_image_thread(self):
        try:
            self.progress_bar.start()
            self.progress_var.set("Starting enhancement...")
            
            input_path = self.input_path.get()
            output_path = self.output_path.get()
            model_name = self.model_name.get()
            scale = self.scale_factor.get()
            
            self.log_message(f"🚀 Starting enhancement...")
            self.log_message(f"Input: {input_path}")
            self.log_message(f"Output: {output_path}")
            self.log_message(f"Model: {model_name}")
            self.log_message(f"Scale: {scale}x")
            
            # Check input file
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            # Check model
            config = self.model_configs[model_name]
            if not os.path.exists(config['model_path']):
                raise FileNotFoundError(f"Model file not found: {config['model_path']}")
            
            self.progress_var.set("Loading image...")
            img = cv2.imread(input_path, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError(f"Could not read image: {input_path}")
            
            self.log_message(f"📷 Image loaded: {img.shape[1]}x{img.shape[0]} pixels")
            
            self.progress_var.set("Loading AI model...")
            upsampler = RealESRGANer(
                scale=config['scale'],
                model_path=config['model_path'],
                model=config['model'],
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=False
            )
            
            self.log_message("🤖 AI model loaded successfully")
            
            self.progress_var.set("Enhancing image... (this may take a while)")
            output, _ = upsampler.enhance(img, outscale=scale)
            
            self.progress_var.set("Saving enhanced image...")
            
            # Create output directory if needed
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            success = cv2.imwrite(output_path, output)
            
            if success:
                self.log_message(f"✅ Enhancement completed!")
                self.log_message(f"Original size: {img.shape[1]}x{img.shape[0]}")
                self.log_message(f"Enhanced size: {output.shape[1]}x{output.shape[0]}")
                self.log_message(f"💾 Saved to: {output_path}")
                
                self.progress_var.set("✅ Enhancement completed successfully!")
                
                result = messagebox.askyesno("Success", 
                                           f"✅ Image enhanced successfully!\n\n"
                                           f"Original: {img.shape[1]}x{img.shape[0]} pixels\n"
                                           f"Enhanced: {output.shape[1]}x{output.shape[0]} pixels\n"
                                           f"Saved to: {output_path}\n\n"
                                           f"Would you like to open the output folder?")
                if result:
                    import subprocess
                    subprocess.Popen(f'explorer /select,"{os.path.abspath(output_path)}"')
            else:
                raise ValueError(f"Failed to save image to: {output_path}")
                
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            self.log_message(error_msg)
            self.progress_var.set("❌ Enhancement failed")
            messagebox.showerror("Enhancement Failed", error_msg)
        
        finally:
            self.progress_bar.stop()
    
    def run(self):
        # Create weights directory if it doesn't exist
        if not os.path.exists('weights'):
            os.makedirs('weights')
            self.log_message("📁 Created 'weights' directory")
        
        self.log_message("🎯 Real-ESRGAN Image Enhancer ready!")
        self.log_message("1. Select an input image")
        self.log_message("2. Choose output location")  
        self.log_message("3. Select appropriate model")
        self.log_message("4. Click 'Enhance Image'")
        
        self.root.mainloop()

def main():
    app = ESRGANImageEnhancer()
    app.run()

if __name__ == "__main__":
    main()