# Standard library imports
import os
import sys
import time
import json
import uuid
import queue
import wave
import base64
import io
import threading
from datetime import datetime, timedelta
from time import time as current_timestamp
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union, Tuple

# Third-party imports
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, scrolledtext, filedialog
except ImportError as e:
    print(f"Error importing tkinter modules: {e}")
    sys.exit(1)

import pyautogui
import numpy as np
import cv2
import pytesseract
import mss
import mss.tools
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
import soundcard as sc
from PIL import Image, ImageTk, ImageGrab, ImageEnhance, ImageFilter
from textblob import TextBlob

# Debug print
print("1. Starting ScreenReader application...")
print("2. All required modules imported successfully")
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect, LangDetectException

# Load the English language model for spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If the model is not found, download it
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Configure Tesseract path - try to find it automatically on Windows
try:
    # Try default installation path first
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # Test if the path is valid
    import subprocess
    subprocess.run([pytesseract.pytesseract.tesseract_cmd, '--version'], 
                  capture_output=True, text=True, check=True)
except (FileNotFoundError, subprocess.CalledProcessError):
    try:
        # Try common alternative path
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
        subprocess.run([pytesseract.pytesseract.tesseract_cmd, '--version'], 
                      capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # If not found, show instructions
        print("\n" + "="*80)
        print("Tesseract OCR is not installed or not in your PATH.")
        print("Please install Tesseract OCR from:")
        print("https://github.com/UB-Mannheim/tesseract/wiki")
        print("""
Installation Steps:
1. Download the installer for Windows
2. Run the installer
3. Make sure to check "Add to PATH" during installation
4. Restart your command prompt/IDE after installation
        """)
        print("="*80 + "\n")

class ScreenOverlay:
    def on_close(self):
        """Handle window close event"""
        # Set flag to stop all threads
        self.running = False
        self.is_capturing = False
        self.is_recording = False
        
        # Wait for recording thread to finish
        if hasattr(self, 'recording_thread') and self.recording_thread:
            if self.recording_thread.is_alive():
                self.recording_thread.join(timeout=1.0)
        
        # Destroy the window
        self.root.quit()
        self.root.destroy()
    
    def __init__(self):
        # Initialize main window
        self.root = tk.Tk()
        self.root.withdraw()  # Hide window until fully initialized
        
        try:
            # Set window attributes
            self.root.title("Screen Data Overlay")
            self.root.attributes('-alpha', 0.7)
            self.root.attributes('-topmost', True)
            self.root.overrideredirect(True)  # Remove window decorations
            
            # Get screen dimensions
            temp_root = tk.Tk()
            self.screen_width = temp_root.winfo_screenwidth()
            self.screen_height = temp_root.winfo_screenheight()
            temp_root.destroy()
            
            # Set window size to full screen
            self.root.geometry(f"{self.screen_width}x{self.screen_height}+0+0")
            
            # Make window click-through
            self.root.attributes("-transparentcolor", "white")
            self.root.config(bg='white')
            
            # Create a canvas that fills the window
            self.canvas = tk.Canvas(self.root, bg='white', highlightthickness=0)
            self.canvas.pack(fill='both', expand=True)
            
            # Initialize notification attributes
            self.notification_label = None
            self.notification_timer = None
            
            # Data storage
            self.captured_data = []  # List to store all capture data
            self.is_capturing = False
            self.running = True
            print("Initialized captured_data list")
            
            # Audio recording settings
            self.recording = False
            self.audio_buffer = []
            self.buffer_lock = threading.Lock()
            self.sample_rate = 44100
            self.channels = 2
            self.system_audio = False  # Default to microphone
            self.recording_thread = None
            
            # Context and history
            self.context_history = []
            self.last_ocr_text = ""
            self.transcription = ""
            
            # Initialize UI elements
            self.setup_ui()
            
            # Show the window after everything is set up
            self.root.deiconify()
            
            # Handle window close event
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
            
            # Start the main loop
            self.update()
            self.root.mainloop()
            
        except Exception as e:
            print(f"Error during initialization: {e}")
            if self.root.winfo_exists():
                self.root.destroy()
            raise
    
    def setup_ui(self):
        """Set up the user interface."""
        # Create a draggable control panel
        self.control_panel = tk.Frame(self.root, bg='#f0f0f0', bd=2, relief='raised')
        self.control_panel.place(x=20, y=20, width=300, height=400)
        
        # Create notification label (floating, not inside control panel)
        self.notification_label = tk.Label(
            self.root, 
            text="", 
            font=('Arial', 12, 'bold'),
            bg='yellow',
            fg='black',
            bd=2,
            relief='solid',
            padx=10,
            pady=5
        )
        self.notification_label.place(relx=0.5, rely=0.1, anchor='center')
        self.notification_label.lift()
        
        # Set a default style to prevent theme errors
        style = ttk.Style()
        style.theme_use('default')
        
        # Make the control panel draggable
        def start_drag(event):
            self.control_panel.startX = event.x
            self.control_panel.startY = event.y

        def stop_drag(event):
            self.control_panel.startX = None
            self.control_panel.startY = None

        def do_drag(event):
            if hasattr(self.control_panel, 'startX'):
                x = self.control_panel.winfo_x() + (event.x - self.control_panel.startX)
                y = self.control_panel.winfo_y() + (event.y - self.control_panel.startY)
                self.control_panel.place(x=x, y=y)

        self.control_panel.bind("<ButtonPress-1>", start_drag)
        self.control_panel.bind("<B1-Motion>", do_drag)
        self.control_panel.bind("<ButtonRelease-1>", stop_drag)
        
        # Title bar with close button
        title_bar = tk.Frame(self.control_panel, bg='#e0e0e0', bd=1, relief='raised')
        title_bar.pack(fill='x')
        
        # Title label
        title_label = tk.Label(
            title_bar, 
            text="Screen Data Overlay", 
            fg='black', 
            bg='#e0e0e0',
            font=('Arial', 10, 'bold')
        )
        title_label.pack(side='left', padx=5, pady=2)
        
        # Close button
        close_btn = tk.Button(
            title_bar, 
            text='×', 
            command=self.on_close, 
            bd=0, 
            fg='black', 
            bg='#e0e0e0',
            activebackground='red',
            activeforeground='white',
            font=('Arial', 12, 'bold'),
            padx=5,
            pady=0
        )
        close_btn.pack(side='right', padx=0, pady=0)
        
        # Make title bar draggable
        def start_drag(event):
            self.control_panel.startX = event.x
            self.control_panel.startY = event.y

        def stop_drag(event):
            self.control_panel.startX = None
            self.control_panel.startY = None

        def do_drag(event):
            if hasattr(self.control_panel, 'startX'):
                x = self.control_panel.winfo_x() + (event.x - self.control_panel.startX)
                y = self.control_panel.winfo_y() + (event.y - self.control_panel.startY)
                self.control_panel.place(x=x, y=y)
        
        title_bar.bind("<ButtonPress-1>", start_drag)
        title_bar.bind("<B1-Motion>", do_drag)
        title_bar.bind("<ButtonRelease-1>", stop_drag)
        
        # Main content frame
        content_frame = tk.Frame(self.control_panel, bg='#f0f0f0')
        content_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Control buttons frame
        btn_frame = tk.Frame(content_frame, bg='#f0f0f0')
        btn_frame.pack(fill='x', pady=5)
        
        # Capture button
        self.capture_btn = ttk.Button(btn_frame, text="Start Capture", command=self.toggle_capture)
        self.capture_btn.pack(fill='x', pady=2)
        
        # View data button
        view_btn = ttk.Button(btn_frame, text="View Captured Data", command=self.show_captured_data)
        view_btn.pack(fill='x', pady=2)
        
        # Clear data button
        clear_btn = ttk.Button(btn_frame, text="Clear All Data", command=self.clear_data)
        clear_btn.pack(fill='x', pady=2)
        
        # Separator
        ttk.Separator(content_frame, orient='horizontal').pack(fill='x', pady=5)
        
        # Audio controls frame
        audio_frame = ttk.LabelFrame(content_frame, text="Audio Controls", padding=5)
        audio_frame.pack(fill='x', pady=5)
        
        # Audio source toggle button
        self.source_btn = ttk.Button(
            audio_frame,
            text="Switch to System Audio",
            command=self.toggle_audio_source
        )
        self.source_btn.pack(fill='x', pady=2)
        
        # Audio toggle button
        self.audio_btn = ttk.Button(
            audio_frame,
            text="Start Audio Capture",
            command=self.toggle_audio_capture
        )
        self.audio_btn.pack(fill='x', pady=2)
        
        # Audio status
        self.audio_status = ttk.Label(audio_frame, text="Audio: Ready")
        self.audio_status.pack(fill='x', pady=2)
        
        # Status frame
        status_frame = ttk.LabelFrame(content_frame, text="Status", padding=5)
        status_frame.pack(fill='x', pady=5)
        
        # Status label
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack()
        
        # Transcription frame
        trans_frame = ttk.LabelFrame(content_frame, text="Transcription", padding=5)
        trans_frame.pack(fill='both', expand=True, pady=5)
        
        # Audio transcription display
        self.transcription_text = scrolledtext.ScrolledText(
            trans_frame, 
            wrap=tk.WORD,
            font=('Arial', 9),
            height=5
        )
        self.transcription_text.pack(fill='both', expand=True)
        
        # Save transcription button
        save_transcript_btn = ttk.Button(
            content_frame,
            text="Save Transcription",
            command=self.save_transcription
        )
        save_transcript_btn.pack(fill='x', pady=2)
        
        # Context display
        context_frame = ttk.LabelFrame(content_frame, text="Context", padding=5)
        context_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.context_display = scrolledtext.ScrolledText(
            context_frame, 
            wrap='word', 
            height=10,
            font=('Consolas', 9)
        )
        self.context_display.pack(fill='both', expand=True)
        
        # Add a clear button
        clear_btn = ttk.Button(
            context_frame,
            text="Clear Context",
            command=self.clear_context_display
        )
        clear_btn.pack(pady=5)
        
        # View all button
        view_all_btn = ttk.Button(btn_frame, 
                                text="View All",
                                command=self.show_captured_data)
        view_all_btn.pack(side='right', fill='x', expand=True, padx=2)
        
    def toggle_capture(self):
        self.is_capturing = not self.is_capturing
        if self.is_capturing:
            self.capture_btn.config(text="Stop Capture")
            self.status_var.set("Status: Capturing...")
        else:
            self.capture_btn.config(text="Start Capture")
            self.status_var.set("Status: Paused")
    
    def clear_data(self):
        if not self.captured_data:
            self.status_var.set("Status: No data to clear")
            return
            
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all captured data?"):
            self.captured_data = []
            self.canvas.delete("all")
            self.status_var.set("Status: Data cleared")
            if hasattr(self, 'screenshot_refs'):
                del self.screenshot_refs
            if hasattr(self, 'detail_refs'):
                del self.detail_refs

    def update_preview(self, screenshot, text):
        """Update the preview panel with the latest screenshot and text."""
        try:
            # Update image preview
            if screenshot:
                # Resize the image to fit in the preview panel
                max_width = 400
                width, height = screenshot.size
                ratio = min(max_width / width, 1.0)  # Don't scale up small images
                new_size = (int(width * ratio), int(height * ratio))
                img = screenshot.resize(new_size, Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(img)
                self.preview_image_label.configure(image=photo)
                self.preview_image_label.image = photo  # Keep a reference
            
            # Update text preview
            self.preview_text.config(state='normal')
            self.preview_text.delete('1.0', tk.END)
            self.preview_text.insert('1.0', text)
            self.preview_text.config(state='disabled')
            
            # Auto-scroll to the bottom
            self.preview_text.see(tk.END)
            
        except Exception as e:
            print(f"Error updating preview: {e}")
    
    def capture_screen_data(self):
        if not self.is_capturing:
            return
            
        try:
            # Get mouse position
            x, y = pyautogui.position()
            
            # Get active window info
            active_window = pyautogui.getActiveWindow()
            window_info = {
                'title': active_window.title if active_window else 'Unknown',
                'position': (active_window.left, active_window.top) if active_window else (0, 0),
                'size': (active_window.width, active_window.height) if active_window else (0, 0)
            }
            
            # Capture the entire screen
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # Primary monitor
                screenshot = sct.grab(monitor)
                
                # Convert to PIL Image
                img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
                
                # Extract text from the entire screen
                screen_text = self.extract_text_from_image(img)
                
                # Also capture a small region around the mouse
                region_img = img.crop((max(0, x-50), max(0, y-50), 
                                    min(img.width, x+50), min(img.height, y+50)))
                
                # Extract text from the region around the mouse
                region_text = self.extract_text_from_image(region_img)
            
            # Store the data
            capture_time = time.time()
            capture_data = {
                'timestamp': capture_time,
                'mouse_position': (x, y),
                'window_info': window_info,
                'screenshot': img,
                'screen_text': screen_text,
                'region_text': region_text,
                'full_screen': True
            }
            self.captured_data.append(capture_data)
            
            # No live preview - will be shown in View All Data
            
            # Draw a marker at the click position
            marker = self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='red', tags='marker')
            
            # Show a notification of the captured text
            preview_text = (screen_text[:47] + '...') if len(screen_text) > 50 else screen_text
            status_msg = "Captured"
            
            if screen_text.strip():
                status_msg = f"{status_msg}: {preview_text}"
            else:
                status_msg = f"{status_msg} (no text found)"
                
            self.status_var.set(status_msg)
            
            # Add to log
            time_str = datetime.fromtimestamp(capture_time).strftime('%H:%M:%S')
            print(f"[{time_str}] {status_msg}")
            
            # Auto-scroll the canvas to show the marker
            self.canvas.xview_moveto(x / self.root.winfo_screenwidth())
            self.canvas.yview_moveto(y / self.root.winfo_screenheight())
                
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
    
    def extract_text_from_image(self, img):
        """Extract text from image using Tesseract OCR with basic preprocessing."""
        try:
            # Ensure Tesseract path is set
            tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
            # Convert to grayscale if needed
            if img.mode != 'L':
                img = img.convert('L')
            
            # Convert to numpy array for OpenCV
            import numpy as np
            img_arr = np.array(img)
            
            # Apply basic preprocessing
            img_arr = cv2.medianBlur(img_arr, 3)  # Reduce noise
            
            # Apply adaptive thresholding
            img_arr = cv2.adaptiveThreshold(
                img_arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Convert back to PIL Image for Tesseract
            processed_img = Image.fromarray(img_arr)
            
            # Try different PSM modes for better results
            for psm in [6, 3, 11]:  # Try different page segmentation modes
                try:
                    text = pytesseract.image_to_string(
                        processed_img,
                        config=f'--psm {psm} --oem 3 -c preserve_interword_spaces=1'
                    )
                    if text and len(text.strip()) > 2:  # If we got some text
                        return text.strip()
                except Exception as e:
                    print(f"Warning: PSM {psm} failed: {e}")
                    continue
                    
            return ""  # Return empty string if all PSMs failed
            
        except Exception as e:
            print(f"Error in OCR processing: {str(e)}")
            return ""
    
    def record_microphone(self):
        """Record audio from microphone in a separate thread using sounddevice"""
        print("Starting microphone recording...")
        print(f"Thread ID: {threading.get_ident()}")
        
        try:
            import sounddevice as sd
            import queue
            
            print("Successfully imported sounddevice module")
            
            # Audio buffer queue
            audio_queue = queue.Queue()
            
            def audio_callback(indata, frames, time, status):
                """This is called for each audio block from the input stream."""
                if status:
                    print(f"Audio status: {status}")
                if self.recording:
                    audio_queue.put(indata.copy())
            
            # Get list of all input devices
            print("Available audio input devices:")
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                print(f"{i}: {dev['name']} (Inputs: {dev['max_input_channels']})")
            
            # Try to find a suitable input device
            input_device = None
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0 and dev['name'] != 'Microsoft Sound Mapper - Input':
                    input_device = i
                    break
            
            if input_device is None:
                error_msg = "No suitable input device found"
                print(error_msg)
                if hasattr(self, 'audio_status'):
                    self.root.after(0, lambda: self.audio_status.config(text=error_msg))
                return
            
            print(f"Using input device: {devices[input_device]['name']}")
            
            # Update status in the UI
            if hasattr(self, 'audio_status'):
                self.root.after(0, lambda: self.audio_status.config(
                    text=f"Recording from {devices[input_device]['name']}..."))
            
            # Start recording
            with sd.InputStream(device=input_device, samplerate=self.sample_rate,
                             channels=min(2, devices[input_device]['max_input_channels']),
                             callback=audio_callback):
                print(f"Started recording with sample rate: {self.sample_rate} Hz")
                chunk_count = 0
                
                while self.recording:
                    try:
                        # Get audio data from the queue
                        data = audio_queue.get(timeout=0.5)
                        chunk_count += 1
                        
                        if data is not None and len(data) > 0:
                            with self.buffer_lock:
                                self.audio_buffer.append(data)
                            
                            # Print progress every 100 chunks
                            if chunk_count % 100 == 0:
                                print(f"Recorded {chunk_count} chunks, buffer size: {len(self.audio_buffer)}")
                                print(f"Latest data shape: {data.shape}, dtype: {data.dtype}", flush=True)
                        
                    except queue.Empty:
                        # No data in queue, but still recording - continue
                        pass
                    except Exception as e:
                        error_msg = f"Error in microphone recording loop: {e}"
                        print(error_msg)
                        
                        # Update status in the UI
                        if hasattr(self, 'audio_status'):
                            self.root.after(0, lambda: self.audio_status.config(text=error_msg))
                        
                        # If we get multiple errors in a row, break the loop
                        if not hasattr(self, 'error_count'):
                            self.error_count = 0
                        self.error_count += 1
                        
                        if self.error_count > 10:  # After 10 errors, give up
                            print("Too many errors, stopping recording")
                            if hasattr(self, 'audio_status'):
                                self.root.after(0, lambda: self.audio_status.config(
                                    text="Too many errors, stopped recording"))
                            break
                        
                        time.sleep(0.1)  # Wait a bit before retrying
            
            print(f"Microphone recording stopped. Recorded {chunk_count} chunks.")
            print(f"Final buffer size: {len(self.audio_buffer) if hasattr(self, 'audio_buffer') else 'N/A'}")
            
        except Exception as e:
            error_msg = f"Error in microphone recording: {str(e)[:200]}"
            print(error_msg)
            import traceback
            traceback.print_exc()  # Print full traceback
            
            if hasattr(self, 'audio_status'):
                self.root.after(0, lambda: self.audio_status.config(text=error_msg))
    
    def record_system_audio(self):
        """Record system audio in a separate thread using sounddevice"""
        print("Starting system audio recording...")
        print(f"Thread ID: {threading.get_ident()}")
        
        try:
            import sounddevice as sd
            import queue
            import numpy as np
            
            print("Successfully imported sounddevice module for system audio")
            
            # Audio buffer queue
            audio_queue = queue.Queue()
            
            def audio_callback(indata, frames, time, status):
                """This is called for each audio block from the input stream."""
                if status:
                    print(f"System audio status: {status}")
                if self.recording:
                    audio_queue.put(indata.copy())
            
            # Get list of all devices
            print("Available audio devices:")
            devices = sd.query_devices()
            
            # Print all devices for debugging
            for i, dev in enumerate(devices):
                print(f"{i}: {dev['name']} (Inputs: {dev['max_input_channels']}, Outputs: {dev['max_output_channels']})")
            
            # Try to find a loopback device (stereo mix, virtual cable, etc.)
            loopback_device = None
            for i, dev in enumerate(devices):
                dev_name = dev['name'].lower()
                # Look for common loopback device names
                if any(term in dev_name for term in ['stereo mix', 'loopback', 'cable', 'virtual', 'voicemeeter']):
                    if dev['max_input_channels'] > 0:
                        loopback_device = i
                        print(f"Found potential loopback device: {dev['name']}")
                        break
            
            if loopback_device is None:
                # If no loopback device found, try to find any output device that can be used as input
                for i, dev in enumerate(devices):
                    if dev['max_output_channels'] > 0 and dev['max_input_channels'] > 0:
                        loopback_device = i
                        print(f"Using output device as input: {dev['name']}")
                        break
            
            if loopback_device is None:
                error_msg = "No suitable loopback device found. Please install a virtual audio cable or similar software."
                print(error_msg)
                if hasattr(self, 'audio_status'):
                    self.root.after(0, lambda: self.audio_status.config(text=error_msg))
                return
            
            device_info = devices[loopback_device]
            print(f"Using audio device: {device_info['name']}")
            
            # Update status in the UI
            if hasattr(self, 'audio_status'):
                self.root.after(0, lambda: self.audio_status.config(
                    text=f"Recording system audio from {device_info['name']}..."))
            
            # Start recording with the selected device
            with sd.InputStream(device=loopback_device, 
                             samplerate=self.sample_rate,
                             channels=min(2, device_info['max_input_channels']),
                             callback=audio_callback):
                print(f"Started recording with sample rate: {self.sample_rate} Hz")
                chunk_count = 0
                
                while self.recording:
                    try:
                        # Get audio data from the queue with a timeout
                        data = audio_queue.get(timeout=0.5)
                        chunk_count += 1
                        
                        if data is not None and len(data) > 0:
                            with self.buffer_lock:
                                self.audio_buffer.append(data)
                            
                            # Print progress every 100 chunks
                            if chunk_count % 100 == 0:
                                print(f"Recorded {chunk_count} chunks, buffer size: {len(self.audio_buffer)}")
                                print(f"Latest data shape: {data.shape}, dtype: {data.dtype}", flush=True)
                        
                    except queue.Empty:
                        # No data in queue, but still recording - continue
                        pass
                    except Exception as e:
                        error_msg = f"Error in system audio recording loop: {e}"
                        print(error_msg)
                        
                        # Update status in the UI
                        if hasattr(self, 'audio_status'):
                            self.root.after(0, lambda: self.audio_status.config(text=error_msg))
                        
                        # If we get multiple errors in a row, break the loop
                        if not hasattr(self, 'error_count'):
                            self.error_count = 0
                        self.error_count += 1
                        
                        if self.error_count > 5:  # After 5 errors, give up
                            print("Too many errors, stopping system audio recording")
                            if hasattr(self, 'audio_status'):
                                self.root.after(0, lambda: self.audio_status.config(
                                    text="Failed to record system audio"))
                            break
                        
                        time.sleep(0.1)  # Wait a bit before retrying
            
            print(f"System audio recording stopped. Recorded {chunk_count} chunks.")
            print(f"Final buffer size: {len(self.audio_buffer) if hasattr(self, 'audio_buffer') else 'N/A'}")
            
        except Exception as e:
            error_msg = f"Error in system audio recording: {str(e)[:200]}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            if hasattr(self, 'audio_status'):
                self.root.after(0, lambda: self.audio_status.config(
                    text="System audio recording failed"))
    
    def process_audio_buffer(self):
        """Process the recorded audio from buffer"""
        with self.buffer_lock:
            if not self.audio_buffer:
                print("No audio data in buffer to process")
                return
                
            print(f"Processing audio buffer with {len(self.audio_buffer)} chunks")
            
            try:
                # Combine audio chunks
                audio_data = np.vstack(self.audio_buffer)
                print(f"Combined audio data shape: {audio_data.shape}")
                
                # Convert stereo to mono if needed
                if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                    print("Converting stereo to mono")
                    audio_data = np.mean(audio_data, axis=1, keepdims=True)
                
                # Calculate duration and show notification
                duration = len(audio_data) / self.sample_rate
                source = "system" if hasattr(self, 'system_audio') and self.system_audio else "microphone"
                self.show_notification(f"🔊 Processing {duration:.1f}s of {source} audio...")
                
                # Normalize to 16-bit PCM if needed
                if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                    print("Normalizing audio data to 16-bit PCM")
                    audio_data = (audio_data * 32767).astype(np.int16)
                
                # Save audio to a temporary WAV file
                temp_file = "temp_audio.wav"
                try:
                    sf.write(temp_file, audio_data, self.sample_rate, 'PCM_16')
                    print(f"Saved audio to {temp_file}, size: {os.path.getsize(temp_file)} bytes")
                except Exception as e:
                    error_msg = f"Error saving audio file: {e}"
                    print(error_msg)
                    self.show_notification("❌ Failed to save audio")
                    return
                
                # Initialize transcription
                self.transcription = ""
                
                # Update UI to show processing status
                self.root.after(0, lambda: self.audio_status.config(
                    text=f"🔊 Processing {duration:.1f}s {source} audio..."))
                
                # Transcribe the audio if we have speech recognition available
                if hasattr(self, 'speech_recognizer'):
                    try:
                        with sr.AudioFile(temp_file) as source:
                            audio_data = self.speech_recognizer.record(source)
                            try:
                                self.transcription = self.speech_recognizer.recognize_google(audio_data)
                                print(f"Transcription: {self.transcription}")
                                
                                # Update the transcription display if it exists
                                if hasattr(self, 'transcription_text'):
                                    self.root.after(0, lambda: self.transcription_text.config(state='normal'))
                                    self.root.after(0, lambda: self.transcription_text.insert('end', 
                                        f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {self.transcription}\n"))
                                    self.root.after(0, lambda: self.transcription_text.see('end'))
                                    self.root.after(0, lambda: self.transcription_text.config(state='disabled'))
                                
                            except sr.UnknownValueError:
                                print("Could not understand audio")
                                self.transcription = "[Could not understand audio]"
                            except sr.RequestError as e:
                                print(f"Could not request results from Google Speech Recognition service; {e}")
                                self.transcription = "[Speech recognition unavailable]"
                                
                    except Exception as e:
                        error_msg = f"Error in speech recognition: {e}"
                        print(error_msg)
                        self.transcription = f"[Error: {str(e)}]"
                        self.show_notification("❌ Speech recognition failed")
                
                # Create context for the audio data
                audio_context = self.add_context_to_data(
                    data_type="audio",
                    raw_data=audio_data,
                    additional_metadata={
                        "sample_rate": self.sample_rate,
                        "source": source,
                        "transcription": self.transcription if self.transcription else "[No speech detected]",
                        "duration_seconds": duration,
                        "timestamp": time.time()
                    }
                )
                
                # Save the contextual data
                self.save_contextual_data(audio_context, audio_data)
                
                # Update UI to show completion
                self.root.after(0, lambda: self.audio_status.config(
                    text=f"✅ Processed {duration:.1f}s {source} audio"))
                
                # Clean up temporary file
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    print(f"Error cleaning up temp file: {e}")
                
                # Clear the audio buffer
                self.audio_buffer = []
                
                print("Successfully processed audio buffer")
                
            except Exception as e:
                error_msg = f"Error processing audio buffer: {e}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                self.show_notification("❌ Error processing audio")
                
                # Update status with error message
                if hasattr(self, 'audio_status'):
                    self.root.after(0, lambda msg=error_msg: self.audio_status.config(text=msg))
            else:
                # If no error, clear any previous error status
                if hasattr(self, 'audio_status'):
                    self.root.after(0, lambda: self.audio_status.config(text=""))
    
    def clear_context_display(self):
        """Clear the context display"""
        if hasattr(self, 'context_display'):
            self.context_display.delete(1.0, 'end')
    
    def add_context_to_data(self, data_type, raw_data, timestamp=None, additional_metadata=None):
        """Add context to captured data based on its type"""
        if timestamp is None:
            timestamp = datetime.datetime.now().isoformat()
        
        context = {
            "timestamp": timestamp,
            "type": data_type,
            "source": "system_audio" if self.system_audio else "microphone",
            "context": {}
        }
        
        # Add additional metadata if provided
        if additional_metadata:
            context.update(additional_metadata)
        
        # Process based on data type
        if data_type == "audio":
            context.update(self._process_audio_context(raw_data))
        elif data_type == "screenshot":
            context.update(self._process_screenshot_context(raw_data))
        elif data_type == "text":
            context.update(self._process_text_context(raw_data))
        
        return context
    
    def _process_audio_context(self, audio_data):
        """Extract context from audio data"""
        context = {
            "duration_seconds": len(audio_data) / self.sample_rate if hasattr(self, 'sample_rate') else None,
            "size_bytes": len(audio_data.tobytes()) if hasattr(audio_data, 'tobytes') else len(str(audio_data)),
            "format": "numpy_array",
            "channels": audio_data.shape[1] if hasattr(audio_data, 'shape') and len(audio_data.shape) > 1 else 1
        }
        
        # Add transcription if available
        if hasattr(self, 'transcription') and self.transcription:
            context["transcription"] = self.transcription
            # Add sentiment analysis
            context["sentiment"] = self._analyze_sentiment(self.transcription)
            # Extract key topics
            context["topics"] = self._extract_topics(self.transcription)
        
        return context

    def _process_screenshot_context(self, screenshot):
        """Extract context from screenshot"""
        context = {
            "resolution": f"{screenshot.width}x{screenshot.height}",
            "format": "PIL.Image" if hasattr(screenshot, 'format') else str(type(screenshot)),
            "size_bytes": len(screenshot.tobytes()) if hasattr(screenshot, 'tobytes') else len(str(screenshot))
        }
        
        # Perform OCR if needed
        if hasattr(self, 'last_ocr_text') and self.last_ocr_text:
            context["detected_text"] = self.last_ocr_text
            context["text_entities"] = self._extract_entities(self.last_ocr_text)
        
        return context

    def _process_text_context(self, text):
        """Extract context from text data"""
        context = {
            "length_chars": len(text),
            "language": self._detect_language(text),
            "contains_questions": "?" in text,
            "contains_links": "http://" in text or "https://" in text
        }
        
        # Add named entity recognition
        entities = self._extract_entities(text)
        if entities:
            context["entities"] = entities
        
        return context

    # Helper methods
    def _analyze_sentiment(self, text):
        """Basic sentiment analysis"""
        try:
            analysis = TextBlob(text)
            return {
                "polarity": analysis.sentiment.polarity,
                "subjectivity": analysis.sentiment.subjectivity,
                "sentiment": "positive" if analysis.sentiment.polarity > 0 else 
                            "negative" if analysis.sentiment.polarity < 0 else "neutral"
            }
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return {"error": str(e)}

    def _extract_entities(self, text):
        """Extract named entities from text"""
        try:
            doc = nlp(text)
            return [{"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char} 
                   for ent in doc.ents]
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return []

    def _extract_topics(self, text, num_topics=3):
        """Extract key topics from text"""
        try:
            vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
            tfidf = vectorizer.fit_transform([text])
            
            # Get top terms
            feature_names = vectorizer.get_feature_names_out()
            sorted_items = tfidf.toarray().argsort()[:, ::-1]
            top_terms = [feature_names[i] for i in sorted_items[0, :num_topics]]
            
            return top_terms
        except Exception as e:
            print(f"Error extracting topics: {e}")
            return []

    def _detect_language(self, text):
        """Detect language of the text"""
        try:
            return detect(text)
        except (LangDetectException, Exception) as e:
            print(f"Error detecting language: {e}")
            return "unknown"
    
    def save_contextual_data(self, context, raw_data=None):
        """Save contextualized data with metadata"""
        # Create a unique ID for this data point
        data_id = str(uuid.uuid4())
        context["id"] = data_id
        
        # Store raw data if provided
        if raw_data is not None:
            # In a real implementation, you might want to save large binary data to disk
            # and just store the path in the context
            if hasattr(raw_data, 'shape') and len(raw_data.shape) > 1:
                context["data_shape"] = raw_data.shape
                context["data_dtype"] = str(raw_data.dtype)
            # For this example, we'll just store a small sample
            context["data_sample"] = str(raw_data[:10]) if hasattr(raw_data, '__getitem__') else str(raw_data)
        
        # Add to history
        if not hasattr(self, 'context_history'):
            self.context_history = []
        self.context_history.append(context)
        
        # Update UI with the new context
        self.update_context_display(context)
        
        # Print to console for debugging
        print(f"Saved context: {json.dumps(context, indent=2, default=str)}")
        
        return context
    
    def show_notification(self, message, duration=3000):
        """Show a temporary notification message"""
        if hasattr(self, 'notification_timer') and self.notification_timer:
            self.root.after_cancel(self.notification_timer)
        
        self.notification_label.config(text=message)
        self.notification_label.lift()
        self.notification_timer = self.root.after(duration, self.clear_notification)
    
    def clear_notification(self):
        """Clear the notification message"""
        if hasattr(self, 'notification_label'):
            self.notification_label.config(text="")
    
    def update_context_display(self, context):
        """Update the UI with the latest context"""
        if not hasattr(self, 'context_display'):
            return
            
        try:
            # Create a summary of the context
            summary = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {context.get('type', 'data').upper()}: "
            
            if context.get('type') == 'audio':
                duration = context.get('duration_seconds', 0)
                source = context.get('source', 'unknown')
                summary += f"🎤 {source.capitalize()} Audio ({duration:.1f}s)"
                
                if 'transcription' in context and context['transcription']:
                    trans = context['transcription']
                    summary += f"\n   🗣️ {trans[:100]}"
                    if len(trans) > 100:
                        summary += "..."
                
                # Show notification for new transcription
                if 'transcription' in context and context['transcription']:
                    self.show_notification(f"🎤 Captured {duration:.1f}s of {source} audio")
                    
            elif context.get('type') == 'screenshot':
                res = context.get('resolution', 'unknown')
                summary += f"📸 Screenshot ({res})"
                
                if 'detected_text' in context and context['detected_text']:
                    text = context['detected_text']
                    summary += f"\n   📝 {text[:100]}"
                    if len(text) > 100:
                        summary += "..."
                
                self.show_notification("📸 Screenshot captured")
                
            elif context.get('type') == 'text':
                text = context.get('text', '')
                summary += f"📋 {text[:200]}"
                if len(text) > 200:
                    summary += "..."
                
                self.show_notification("📋 Text captured")
            
            # Add to the display with proper formatting
            self.context_display.config(state='normal')
            self.context_display.insert('end', summary + '\n\n')
            self.context_display.see('end')
            self.context_display.config(state='disabled')
            
            # Also update the transcription display if it's audio with transcription
            if context.get('type') == 'audio' and 'transcription' in context and context['transcription']:
                self.transcription_text.config(state='normal')
                self.transcription_text.insert('end', f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {context['transcription']}\n")
                self.transcription_text.see('end')
                self.transcription_text.config(state='disabled')
                
        except Exception as e:
            print(f"Error updating context display: {e}")
    
    def toggle_audio_source(self):
        """Toggle between microphone and system audio sources"""
        try:
            self.system_audio = not self.system_audio
            source = "System Audio" if self.system_audio else "Microphone"
            self.source_btn.config(text=f"Switch to {'Microphone' if self.system_audio else 'System Audio'}")
            self.audio_status.config(text=f"Audio: {source} (Ready)")
            print(f"Switched to {source}")
            
            # If currently recording, restart with the new source
            if self.recording:
                self.toggle_audio_capture()  # Stop current recording
                time.sleep(0.5)  # Small delay
                self.toggle_audio_capture()  # Start new recording
        except Exception as e:
            error_msg = f"Error switching audio source: {e}"
            print(error_msg)
            self.audio_status.config(text=error_msg)
    
    def toggle_audio_capture(self):
        """Toggle audio recording on/off"""
        try:
            if not self.recording:
                print(f"Starting {'system audio' if self.system_audio else 'microphone'} recording...")
                print(f"Current audio source: {'System' if self.system_audio else 'Microphone'}")
                
                # Clear previous buffer and reset error count
                with self.buffer_lock:
                    self.audio_buffer = []
                self.error_count = 0
                
                # Start recording in a new thread
                self.recording = True
                recording_target = self.record_system_audio if self.system_audio else self.record_microphone
                print(f"Recording target: {recording_target.__name__}")
                
                self.recording_thread = threading.Thread(
                    target=recording_target,
                    daemon=True
                )
                self.recording_thread.start()
                
                # Verify thread started
                if not self.recording_thread.is_alive():
                    raise Exception("Failed to start recording thread")
                    
                self.audio_btn.config(text="Stop Audio Capture")
                status_text = f"Audio: Recording from {'System' if self.system_audio else 'Microphone'}..."
                self.audio_status.config(text=status_text)
                print(f"Started {'system audio' if self.system_audio else 'microphone'} recording")
                print(f"Thread {self.recording_thread.ident} started")
            else:
                print("Stopping recording...")
                # Stop recording
                self.recording = False
                
                if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
                    print("Waiting for recording thread to finish...")
                    self.recording_thread.join(timeout=2.0)
                    if self.recording_thread.is_alive():
                        print("Warning: Recording thread did not stop gracefully")
                
                self.audio_btn.config(text="Start Audio Capture")
                status_text = f"Audio: {'System Audio' if self.system_audio else 'Microphone'} (Ready)"
                self.audio_status.config(text=status_text)
                print("Stopped audio recording")
                
                # Process the recorded audio if we have any data
                if hasattr(self, 'audio_buffer') and self.audio_buffer:
                    print(f"Processing {len(self.audio_buffer)} audio chunks...")
                    threading.Thread(target=self.process_audio_buffer, daemon=True).start()
                else:
                    print("No audio data was recorded")
        except Exception as e:
            error_msg = f"Error toggling audio capture: {e}"
            print(error_msg)
            self.audio_status.config(text=error_msg)
            self.recording = False
            if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=1.0)
    
    def show_captured_data(self):
        if not self.captured_data:
            messagebox.showinfo("No Data", "No data has been captured yet.")
            return
        
        # Create a new window
        data_window = tk.Toplevel(self.root)
        data_window.title("Captured Data")
        data_window.geometry("1200x800")
        
        # Create a notebook for tabs
        notebook = ttk.Notebook(data_window)
        notebook.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Tab 4: Audio Transcriptions
        audio_frame = ttk.Frame(notebook)
        notebook.add(audio_frame, text="Audio Transcriptions")
        
        # Audio controls
        audio_controls = ttk.Frame(audio_frame)
        audio_controls.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(audio_controls, text="Export All Transcriptions", 
                  command=self.export_all_transcriptions).pack(side='left', padx=5)
        
        # Audio transcription display with timestamps
        self.audio_transcription_text = scrolledtext.ScrolledText(
            audio_frame, wrap=tk.WORD, font=('Consolas', 10))
        self.audio_transcription_text.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Load existing transcriptions if any
        if hasattr(self, 'transcription_text') and self.transcription_text.get('1.0', 'end-1c'):
            self.audio_transcription_text.insert('1.0', self.transcription_text.get('1.0', 'end-1c'))
        
        # Tab 1: Events List of events with text preview
        events_frame = ttk.Frame(notebook)
        notebook.add(events_frame, text="Events")
        
        # Treeview for events with more columns
        columns = ("#", "Time", "X", "Y", "Window Title", "Text Preview")
        tree = ttk.Treeview(events_frame, columns=columns, show='headings')
        
        # Define headings and column widths
        col_widths = [40, 80, 50, 50, 200, 300]
        for col, width in zip(columns, col_widths):
            tree.heading(col, text=col)
            tree.column(col, width=width, anchor='w')
        
        # Add data to treeview
        for idx, data in enumerate(self.captured_data, 1):
            try:
                # Handle both float and datetime objects
                if 'timestamp' in data and data['timestamp'] is not None:
                    if isinstance(data['timestamp'], (int, float)):
                            timestamp = datetime.fromtimestamp(float(data['timestamp'])).strftime('%H:%M:%S')
                    else:
                        timestamp = str(data['timestamp'])[:8]  # Just take the time part if it's already a string
                else:
                    timestamp = 'N/A'
            except (KeyError, TypeError) as e:
                print(f"Error processing timestamp: {e}")
                timestamp = 'N/A'
            x, y = data['mouse_position']
            window_title = data['window_info']['title'][:30] + '...' if len(data['window_info']['title']) > 30 else data['window_info']['title']
            
            # Get text preview (from region if available, otherwise from screen)
            text_preview = data.get('region_text', data.get('screen_text', ''))
            text_preview = text_preview.replace('\n', ' ').strip()
            text_preview = (text_preview[:47] + '...') if len(text_preview) > 50 else text_preview
            
            tree.insert('', 'end', values=(
                idx, 
                timestamp, 
                f"{x},{y}", 
                f"{data['window_info']['size'][0]}x{data['window_info']['size'][1]}",
                window_title,
                text_preview
            ), tags=('click',))
        
        # Add scrollbars
        vsb = ttk.Scrollbar(events_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(events_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid layout
        tree.grid(column=0, row=0, sticky='nsew')
        vsb.grid(column=1, row=0, sticky='ns')
        hsb.grid(column=0, row=1, sticky='ew')
        
        # Configure grid weights
        events_frame.grid_columnconfigure(0, weight=1)
        events_frame.grid_rowconfigure(0, weight=1)
        
        # Tab 2: Extracted Text
        text_frame = ttk.Frame(notebook)
        notebook.add(text_frame, text="Extracted Text")
        
        text_area = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, width=80, height=30)
        text_area.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Add extracted text
        all_text = []
        for idx, data in enumerate(self.captured_data, 1):
            if 'screen_text' in data and data['screen_text']:
                all_text.append(f"--- Capture #{idx} ---")
                all_text.append(f"Time: {datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
                all_text.append(f"Position: {data['mouse_position']}")
                all_text.append(f"Window: {data['window_info']['title']}")
                all_text.append("Extracted Text:")
                all_text.append(data['screen_text'])
                all_text.append("\n" + "-"*50 + "\n")
        
        text_area.insert(tk.END, '\n'.join(all_text) if all_text else "No text extracted yet.")
        text_area.config(state='disabled')
        
        # Tab 3: Screenshots & Text
        if any('screenshot' in data for data in self.captured_data if data.get('screenshot')):
            screenshots_frame = ttk.Frame(notebook)
            notebook.add(screenshots_frame, text="Screenshots & Text")
            
            # Create a PanedWindow for resizable split view
            paned = ttk.PanedWindow(screenshots_frame, orient=tk.HORIZONTAL)
            paned.pack(fill=tk.BOTH, expand=True)
            
            # Left side: List of captures
            list_frame = ttk.Frame(paned, width=200)
            paned.add(list_frame, weight=1)
            
            # Right side: Details view
            detail_frame = ttk.Frame(paned)
            paned.add(detail_frame, weight=3)
            
            # List of captures
            listbox = tk.Listbox(list_frame, selectmode=tk.SINGLE)
            scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=listbox.yview)
            listbox.config(yscrollcommand=scrollbar.set)
            
            # Add items to listbox
            for idx, data in enumerate(self.captured_data, 1):
                time_str = datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S') if 'timestamp' in data else 'No timestamp'
                preview = data.get('region_text', data.get('screen_text', ''))[:30]
                listbox.insert(tk.END, f"{idx}. {time_str} - {preview}...")
            
            # Pack listbox and scrollbar
            listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Detail view
            detail_canvas = tk.Canvas(detail_frame)
            detail_scrollbar = ttk.Scrollbar(detail_frame, orient="vertical", command=detail_canvas.yview)
            detail_content = ttk.Frame(detail_canvas)
            
            detail_canvas.create_window((0, 0), window=detail_content, anchor='nw')
            detail_canvas.configure(yscrollcommand=detail_scrollbar.set)
            
            def on_mousewheel(event):
                detail_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            
            detail_canvas.bind_all("<MouseWheel>", on_mousewheel)
            
            def on_frame_configure(event):
                detail_canvas.configure(scrollregion=detail_canvas.bbox("all"))
            
            detail_content.bind("<Configure>", on_frame_configure)
            
            # Function to update detail view
            def show_capture_details(event=None):
                selection = listbox.curselection()
                if not selection:
                    return
                    
                idx = selection[0]
                data = self.captured_data[idx]
                
                # Clear previous content
                for widget in detail_content.winfo_children():
                    widget.destroy()
                
                # Show screenshot
                screenshot = data['screenshot']
                
                # Resize for display (maintain aspect ratio)
                max_width = 600
                max_height = 400
                ratio = min(max_width/screenshot.width, max_height/screenshot.height)
                new_size = (int(screenshot.width * ratio), int(screenshot.height * ratio))
                display_img = screenshot.resize(new_size, Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(display_img)
                
                # Keep reference to avoid garbage collection
                if not hasattr(self, 'detail_photos'):
                    self.detail_photos = []
                self.detail_photos.append(photo)
                
                img_label = ttk.Label(detail_content, image=photo)
                img_label.image = photo
                img_label.pack(pady=10, padx=10)
                
                # Show mouse position
                x, y = data['mouse_position']
                pos_label = ttk.Label(detail_content, 
                                    text=f"Mouse Position: ({x}, {y}) | Window: {data['window_info']['title']}")
                pos_label.pack(pady=5)
                
                # Show extracted text
                text_frame = ttk.LabelFrame(detail_content, text="Extracted Text", padding=10)
                text_frame.pack(fill='both', expand=True, padx=10, pady=5)
                
                text_area = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, width=80, height=10)
                text_area.pack(fill='both', expand=True)
                
                # Add extracted text
                if 'screen_text' in data and data['screen_text']:
                    text_area.insert(tk.END, data['screen_text'])
                text_area.config(state='disabled')
                
                # Add buttons
                btn_frame = ttk.Frame(detail_content)
                btn_frame.pack(fill='x', pady=5)
                
                save_text_btn = ttk.Button(btn_frame, text="Save Text", 
                                          command=lambda d=data: self.save_text(d))
                save_text_btn.pack(side='left', padx=5)
                
                copy_btn = ttk.Button(btn_frame, text="Copy Text", 
                                    command=lambda t=data.get('screen_text', ''): 
                                    self.root.clipboard_clear() or self.root.clipboard_append(t))
                copy_btn.pack(side='left', padx=5)
                
                save_img_btn = ttk.Button(btn_frame, text="Save Screenshot", 
                                        command=lambda d=data: self.save_screenshot(d))
                save_img_btn.pack(side='right', padx=5)
            
            # Bind selection change
            listbox.bind('<<ListboxSelect>>', show_capture_details)
            
            # Show first capture by default
            if len(self.captured_data) > 0:
                listbox.selection_set(0)
                listbox.event_generate('<<ListboxSelect>>')
            
            # Pack detail view
            detail_canvas.pack(side='left', fill='both', expand=True)
            detail_scrollbar.pack(side='right', fill='y')
    
    def save_screenshot(self, data):
        """Save screenshot to file."""
        if 'screenshot' not in data or not data['screenshot']:
            messagebox.showerror("Error", "No screenshot available to save")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            initialfile=f"screenshot_{int(time.time())}.png"
        )
        
        if filename:
            try:
                data['screenshot'].save(filename)
                messagebox.showinfo("Success", f"Screenshot saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save screenshot: {str(e)}")
    
    def save_text(self, data):
        """Save extracted text to file."""
        if 'screen_text' not in data or not data['screen_text']:
            messagebox.showerror("Error", "No text available to save")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=f"extracted_text_{int(time.time())}.txt"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(data['screen_text'])
                messagebox.showinfo("Success", f"Text saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save text: {str(e)}")
        
        # Event binding for treeview selection
        def on_tree_select(event):
            selected_item = tree.selection()[0]
            item_values = tree.item(selected_item, 'values')
            if item_values:
                idx = int(item_values[0]) - 1
                if 0 <= idx < len(self.captured_data):
                    data = self.captured_data[idx]
                    self.show_click_details(data)
        
        tree.bind('<<TreeviewSelect>>', on_tree_select)
    
    def show_click_details(self, data):
        details_window = tk.Toplevel(self.root)
        details_window.title("Click Details")
        details_window.geometry("400x300")
        
        text = f"""
        Time: {dt.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}
        
        Mouse Position: {data['mouse_position']}
        
        Window Information:
          Title: {data['window_info']['title']}
          Position: {data['window_info']['position']}
          Size: {data['window_info']['size']}
        """
        
        text_widget = scrolledtext.ScrolledText(details_window, wrap=tk.WORD, width=50, height=15)
        text_widget.pack(fill='both', expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, text)
        text_widget.config(state='disabled')
        
        # Show screenshot if available
        if 'screenshot' in data and data['screenshot']:
            try:
                screenshot = data['screenshot']
                photo = ImageTk.PhotoImage(screenshot)
                
                # Keep a reference
                if not hasattr(self, 'detail_refs'):
                    self.detail_refs = []
                self.detail_refs.append(photo)
                
                label = ttk.Label(details_window, image=photo)
                label.image = photo  # Keep reference
                
                # Resize for display (maintain aspect ratio)
                max_width = 600
                max_height = 400
                ratio = min(max_width/screenshot.width, max_height/screenshot.height)
                new_size = (int(screenshot.width * ratio), int(screenshot.height * ratio))
                display_img = screenshot.resize(new_size, Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(display_img)
                
                # Keep reference to avoid garbage collection
                if not hasattr(self, 'detail_photos'):
                    self.detail_photos = []
                self.detail_photos.append(photo)
                
                img_label = ttk.Label(details_window, image=photo)
                img_label.image = photo
                img_label.pack(pady=10, padx=10)
                
                # Show mouse position
                x, y = data['mouse_position']
                pos_label = ttk.Label(details_window, 
                                    text=f"Mouse Position: ({x}, {y}) | Window: {data['window_info']['title']}")
                pos_label.pack(pady=5)
                
                # Show extracted text
                text_frame = ttk.LabelFrame(details_window, text="Extracted Text", padding=10)
                text_frame.pack(fill='both', expand=True, padx=10, pady=5)
                
                text_area = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, width=80, height=10)
                text_area.pack(fill='both', expand=True)
                
                # Add extracted text
                if 'screen_text' in data and data['screen_text']:
                    text_area.insert(tk.END, data['screen_text'])
                text_area.config(state='disabled')
                
                # Add buttons
                btn_frame = ttk.Frame(details_window)
                btn_frame.pack(fill='x', pady=5)
                
                save_text_btn = ttk.Button(btn_frame, text="Save Text", 
                                      command=lambda d=data: self.save_text(d))
                save_text_btn.pack(side='left', padx=5)
                
                copy_btn = ttk.Button(btn_frame, text="Copy Text", 
                                    command=lambda t=data.get('screen_text', ''): 
                                    self.root.clipboard_clear() or self.root.clipboard_append(t))
                copy_btn.pack(side='left', padx=5)
                
                save_img_btn = ttk.Button(btn_frame, text="Save Screenshot", 
                                        command=lambda d=data: self.save_screenshot(d))
                save_img_btn.pack(side='right', padx=5)
                
            except Exception as e:
                print(f"Error showing click details: {e}")
                temp_filename = temp_file.name
            
            try:
                # Save as WAV file
                sf.write(temp_filename, audio_data, self.sample_rate)
                print(f"Saved audio to {temp_filename}")
                
                # Process the audio file in a separate thread
                threading.Thread(
                    target=self._save_and_process_audio,
                    args=(audio_data, temp_filename),
                    daemon=True
                ).start()
                
            except Exception as e:
                error_msg = f"Error processing audio: {e}"
                print(error_msg)
                if hasattr(self, 'audio_status'):
                    self.root.after(0, lambda: self.audio_status.config(
                        text=error_msg))
                # Clean up the temp file if it exists
                if temp_filename and os.path.exists(temp_filename):
                    try:
                        os.unlink(temp_filename)
                    except Exception as cleanup_error:
                        print(f"Error cleaning up temp file: {cleanup_error}")
                if hasattr(self, 'audio_status'):
                    self.root.after(0, lambda: self.audio_status.config(text=error_msg))
        
        print("System audio recording stopped")
    
    def toggle_audio_source(self):
        """Toggle between microphone and system audio sources"""
        try:
            self.system_audio = not self.system_audio
            source = "System Audio" if self.system_audio else "Microphone"
            self.source_btn.config(text=f"Switch to {'Microphone' if self.system_audio else 'System Audio'}")
            self.audio_status.config(text=f"Audio: {source} (Ready)")
            print(f"Switched to {source}")
            
            # If currently recording, restart with the new source
            if self.recording:
                self.toggle_audio_capture()  # Stop current recording
                time.sleep(0.5)  # Small delay
                self.toggle_audio_capture()  # Start new recording
        except Exception as e:
            error_msg = f"Error switching audio source: {e}"
            print(error_msg)
            self.audio_status.config(text=error_msg)
    
    def toggle_audio_capture(self):
        """Toggle audio recording on/off"""
        try:
            if not self.recording:
                print(f"Starting {'system audio' if self.system_audio else 'microphone'} recording...")
                print(f"Current audio source: {'System' if self.system_audio else 'Microphone'}")
                
                # Clear previous buffer and reset error count
                with self.buffer_lock:
                    self.audio_buffer = []
                self.error_count = 0
                
                # Start recording in a new thread
                self.recording = True
                recording_target = self.record_system_audio if self.system_audio else self.record_microphone
                print(f"Recording target: {recording_target.__name__}")
                
                self.recording_thread = threading.Thread(
                    target=recording_target,
                    daemon=True
                )
                self.recording_thread.start()
                
                # Verify thread started
                if not self.recording_thread.is_alive():
                    raise Exception("Failed to start recording thread")
                    
                self.audio_btn.config(text="Stop Audio Capture")
                status_text = f"Audio: Recording from {'System' if self.system_audio else 'Microphone'}..."
                self.audio_status.config(text=status_text)
                print(f"Started {'system audio' if self.system_audio else 'microphone'} recording")
                print(f"Thread {self.recording_thread.ident} started")
            else:
                print("Stopping recording...")
                # Stop recording
                self.recording = False
                
                if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
                    print("Waiting for recording thread to finish...")
                    self.recording_thread.join(timeout=2.0)
                    if self.recording_thread.is_alive():
                        print("Warning: Recording thread did not stop gracefully")
                
                self.audio_btn.config(text="Start Audio Capture")
                status_text = f"Audio: {'System Audio' if self.system_audio else 'Microphone'} (Ready)"
                self.audio_status.config(text=status_text)
                print("Stopped audio recording")
                
                # Process the recorded audio if we have any data
                if hasattr(self, 'audio_buffer') and self.audio_buffer:
                    print(f"Processing {len(self.audio_buffer)} audio chunks...")
                    threading.Thread(target=self.process_audio_buffer, daemon=True).start()
                else:
                    print("No audio data was recorded")
        except Exception as e:
            error_msg = f"Error toggling audio capture: {e}"
            print(error_msg)
            self.audio_status.config(text=error_msg)
            self.recording = False
            if hasattr(self, 'recording_thread') and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=1.0)
    
    def _save_and_process_audio(self, audio_data, temp_filename):
        """Helper method to save and process audio data"""
        print(f"Saving audio to {temp_filename}")
        try:
            with sf.SoundFile(temp_filename, mode='w', 
                           samplerate=self.sample_rate, 
                           channels=self.channels) as file:
                file.write(audio_data)
            print("Audio file saved successfully")
            
            # Verify file was written
            if os.path.exists(temp_filename) and os.path.getsize(temp_filename) > 0:
                print("Audio file exists and has content")
                # Transcribe the audio
                self.transcribe_audio(temp_filename)
            else:
                print("Error: Audio file is empty or not created")
                if hasattr(self, 'audio_status'):
                    self.audio_status.config(text="Audio: Failed to save recording")
                
        except Exception as e:
            print(f"Error saving audio file: {e}")
            if hasattr(self, 'audio_status'):
                self.audio_status.config(text=f"Audio Error: {str(e)}")
        finally:
            # Clean up
            try:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                    print("Temporary audio file removed")
            except Exception as e:
                print(f"Error removing temp file: {e}")
    
    def transcribe_audio(self, audio_file):
        """Transcribe the recorded audio file"""
        print(f"Starting transcription of {audio_file}")
        try:
            # Verify file exists and has content
            if not os.path.exists(audio_file) or os.path.getsize(audio_file) == 0:
                raise Exception("Audio file is empty or doesn't exist")
                
            r = sr.Recognizer()
            with sr.AudioFile(audio_file) as source:
                print("Reading audio file...")
                audio_data = r.record(source)
                print("Transcribing audio...")
                text = r.recognize_google(audio_data)
                print(f"Transcription successful: {text[:50]}...")
                
                # Add timestamp and format the entry
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                formatted_entry = f"[Audio - {timestamp}]\n{text}\n" + ("-"*80) + "\n\n"
                
                # Update the UI in a thread-safe way
                def update_ui():
                    try:
                        if hasattr(self, 'transcription_text') and self.transcription_text.winfo_exists():
                            self.transcription_text.insert(tk.END, formatted_entry)
                            self.transcription_text.see(tk.END)
                        if hasattr(self, 'audio_transcription_text') and self.audio_transcription_text.winfo_exists():
                            self.audio_transcription_text.insert(tk.END, formatted_entry)
                            self.audio_transcription_text.see(tk.END)
                        if hasattr(self, 'audio_status') and self.audio_status.winfo_exists():
                            self.audio_status.config(text=f"Audio: Transcribed at {timestamp}")
                    except Exception as ui_error:
                        print(f"UI update error: {ui_error}")
                
                self.root.after(0, update_ui)
                
                # Return the transcription for further processing if needed
                return {
                    'timestamp': timestamp,
                    'text': text,
                    'formatted': formatted_entry
                }
                
        except sr.UnknownValueError:
            error_msg = "Audio Error: Could not understand audio"
        except sr.RequestError as e:
            error_msg = f"Audio Error: Could not request results; {e}"
        except Exception as e:
            error_msg = f"Audio Error: {str(e)}"
            
        print(error_msg)
        if hasattr(self, 'audio_status') and self.audio_status.winfo_exists():
            self.audio_status.config(text=error_msg)
        return None
    
    def export_all_transcriptions(self):
        """Export all transcriptions to a file"""
        if not hasattr(self, 'transcription_text') or not self.transcription_text.get("1.0", tk.END).strip():
            messagebox.showwarning("Warning", "No transcriptions available to export")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"), 
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ],
            initialfile=f"transcriptions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        if filename:
            try:
                content = self.transcription_text.get("1.0", tk.END)
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("=== Audio Transcriptions ===\n")
                    f.write(f"Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(content)
                
                messagebox.showinfo("Success", f"All transcriptions exported to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export transcriptions: {str(e)}")
    
    def save_transcription(self):
        """Save the current transcription to a file"""
        if not hasattr(self, 'transcription_text') or not self.transcription_text.get("1.0", tk.END).strip():
            messagebox.showwarning("Warning", "No transcription to save")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"), 
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ],
            initialfile=f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("=== Audio Transcription ===\n")
                    f.write(f"Recorded on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(self.transcription_text.get("1.0", tk.END))
                messagebox.showinfo("Success", f"Transcription saved to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save transcription: {str(e)}")
    
    def update(self):
        if not self.running:
            return
            
        try:
            self.capture_screen_data()
            if hasattr(self, 'root') and self.running:  # Check if root exists and app is running
                self.root.after(100, self.update)  # Update every 100ms
        except Exception as e:
            print(f"Error in update loop: {e}")
            if hasattr(self, 'root') and self.running:
                self.root.after(100, self.update)
    
    def show_captured_data(self):
        """Show a professional interface for viewing all captured data."""
        print("Attempting to show captured data...")
        print(f"hasattr('captured_data'): {hasattr(self, 'captured_data')}")
        if hasattr(self, 'captured_data'):
            print(f"Length of captured_data: {len(self.captured_data)}")
            
        if not hasattr(self, 'captured_data') or not self.captured_data:
            error_msg = "No capture data available. Please capture some data first."
            print(error_msg)
            if hasattr(self, 'status_var'):
                self.status_var.set(error_msg)
            else:
                print("Warning: status_var not available to show error message")
            return
            
        try:
            # Create a new window for viewing captured data
            view_window = tk.Toplevel(self.root)
            view_window.title("Captured Data")
            view_window.geometry("1200x800")
            
            # Create a notebook for tabs
            notebook = ttk.Notebook(view_window)
            notebook.pack(fill='both', expand=True, padx=5, pady=5)
            
            # Tab 1: List View with Preview
            list_frame = ttk.Frame(notebook)
            notebook.add(list_frame, text="List View")
            
            # Split view with list on left and preview on right
            paned = ttk.PanedWindow(list_frame, orient=tk.HORIZONTAL)
            paned.pack(fill='both', expand=True, padx=5, pady=5)
            
            # Left panel: List of captures
            list_container = ttk.Frame(paned)
            paned.add(list_container, weight=1)
            
            # Add search/filter
            search_frame = ttk.Frame(list_container)
            search_frame.pack(fill='x', pady=5)
            
            ttk.Label(search_frame, text="Search:").pack(side='left', padx=5)
            search_var = tk.StringVar()
            search_entry = ttk.Entry(search_frame, textvariable=search_var)
            search_entry.pack(side='left', fill='x', expand=True, padx=5)
            
            # Listbox with scrollbar
            listbox_frame = ttk.Frame(list_container)
            listbox_frame.pack(fill='both', expand=True)
            
            scrollbar = ttk.Scrollbar(listbox_frame)
            scrollbar.pack(side='right', fill='y')
            
            self.capture_list = tk.Listbox(
                listbox_frame, 
                yscrollcommand=scrollbar.set,
                selectmode='single',
                font=('Courier', 10)
            )
            self.capture_list.pack(side='left', fill='both', expand=True)
            scrollbar.config(command=self.capture_list.yview)
            
            # Right panel: Preview
            preview_frame = ttk.Frame(paned)
            paned.add(preview_frame, weight=2)
            
            # Preview tabs
            preview_notebook = ttk.Notebook(preview_frame)
            preview_notebook.pack(fill='both', expand=True)
            
            # Image preview tab
            image_tab = ttk.Frame(preview_notebook)
            preview_notebook.add(image_tab, text="Screenshot")
            
            self.preview_image_label = ttk.Label(image_tab)
            self.preview_image_label.pack(pady=10)
            
            # Text preview tab
            text_tab = ttk.Frame(preview_notebook)
            preview_notebook.add(text_tab, text="Extracted Text")
            
            self.preview_text = scrolledtext.ScrolledText(
                text_tab, 
                wrap=tk.WORD,
                font=('Courier', 10)
            )
            self.preview_text.pack(fill='both', expand=True, padx=5, pady=5)
            
            # Metadata tab
            meta_tab = ttk.Frame(preview_notebook)
            preview_notebook.add(meta_tab, text="Metadata")
            
            self.meta_text = scrolledtext.ScrolledText(
                meta_tab,
                wrap=tk.WORD,
                font=('Courier', 10),
                height=10
            )
            self.meta_text.pack(fill='both', expand=True, padx=5, pady=5)
            
            # Action buttons
            btn_frame = ttk.Frame(preview_frame)
            btn_frame.pack(fill='x', pady=5)
            
            ttk.Button(btn_frame, text="Save Image", command=self.save_current_image).pack(side='left', padx=5)
            ttk.Button(btn_frame, text="Copy Text", command=self.copy_current_text).pack(side='left', padx=5)
            ttk.Button(btn_frame, text="Export All", command=self.export_all_data).pack(side='right', padx=5)
            
            # Populate the list
            self.populate_capture_list()
            
            # Bind selection
            self.capture_list.bind('<<ListboxSelect>>', self.on_capture_select)
            
            # Bind search
            search_var.trace('w', lambda *args: self.on_search(search_var.get()))
            
            # Select first item
            if self.captured_data:
                self.capture_list.selection_set(0)
                self.capture_list.see(0)
                self.on_capture_select()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open captured data: {str(e)}")
    
    def populate_capture_list(self, filter_text=''):
        """Populate the capture list with filtered items."""
        if not hasattr(self, 'capture_list'):
            return
            
        self.capture_list.delete(0, tk.END)
        filter_text = filter_text.lower()
        
        for idx, data in enumerate(self.captured_data, 1):
            # Format the list item
            timestamp = datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            preview = data.get('screen_text', '')[:50] + '...' if 'screen_text' in data else '(No text)'
            item_text = f"{idx:3d}. {timestamp} - {preview}"
            
            # Apply filter
            if filter_text in item_text.lower() or not filter_text:
                self.capture_list.insert(tk.END, item_text)
    
    def on_capture_select(self, event=None):
        """Handle selection of a capture from the list."""
        if not hasattr(self, 'capture_list') or not hasattr(self, 'captured_data'):
            return
            
        selection = self.capture_list.curselection()
        if not selection:
            return
            
        idx = selection[0]
        data = self.captured_data[idx]
        
        # Update image preview
        if 'screenshot' in data and data['screenshot']:
            img = data['screenshot']
            # Resize to fit while maintaining aspect ratio
            max_size = (800, 600)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            self.preview_image_label.configure(image=photo)
            self.preview_image_label.image = photo
        
        # Update text preview
        if hasattr(self, 'preview_text'):
            self.preview_text.config(state='normal')
            self.preview_text.delete('1.0', tk.END)
            self.preview_text.insert('1.0', data.get('screen_text', 'No text extracted'))
            self.preview_text.config(state='disabled')
        
        # Update metadata
        if hasattr(self, 'meta_text'):
            meta_info = []
            meta_info.append(f"Timestamp: {datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
            meta_info.append(f"Mouse Position: {data['mouse_position']}")
            meta_info.append(f"Window Title: {data['window_info'].get('title', 'N/A')}")
            meta_info.append(f"Window Size: {data['window_info'].get('size', ('N/A', 'N/A'))}")
            meta_info.append(f"Capture Type: {'Full Screen' if data.get('full_screen', False) else 'Region'}")
            
            self.meta_text.config(state='normal')
            self.meta_text.delete('1.0', tk.END)
            self.meta_text.insert('1.0', '\n'.join(meta_info))
            self.meta_text.config(state='disabled')
    
    def on_search(self, search_text):
        """Handle search/filter changes."""
        self.populate_capture_list(search_text)
    
    def save_current_image(self):
        """Save the currently selected image to a file."""
        if not hasattr(self, 'capture_list') or not hasattr(self, 'captured_data'):
            return
            
        selection = self.capture_list.curselection()
        if not selection:
            return
            
        idx = selection[0]
        data = self.captured_data[idx]
        
        if 'screenshot' not in data or not data['screenshot']:
            messagebox.showwarning("No Image", "No screenshot available to save.")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                data['screenshot'].save(file_path)
                messagebox.showinfo("Success", f"Image saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def copy_current_text(self):
        """Copy the current text to clipboard."""
        if hasattr(self, 'preview_text'):
            self.root.clipboard_clear()
            self.root.clipboard_append(self.preview_text.get('1.0', 'end-1c'))
            self.status_var.set("Text copied to clipboard")
    
    def export_all_data(self):
        """Export all captured data to a folder."""
        if not hasattr(self, 'captured_data') or not self.captured_data:
            messagebox.showwarning("No Data", "No capture data to export.")
            return
            
        folder_path = filedialog.askdirectory()
        if not folder_path:
            return
            
        try:
            # Create timestamped subfolder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_folder = os.path.join(folder_path, f"capture_export_{timestamp}")
            os.makedirs(export_folder, exist_ok=True)
            
            # Export each capture
            for idx, data in enumerate(self.captured_data, 1):
                # Save screenshot if available
                if 'screenshot' in data and data['screenshot']:
                    img_path = os.path.join(export_folder, f"capture_{idx:03d}.png")
                    data['screenshot'].save(img_path)
                
                # Save text
                text_path = os.path.join(export_folder, f"capture_{idx:03d}.txt")
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(f"Timestamp: {datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}\\n")
                    f.write(f"Mouse Position: {data['mouse_position']}\\n")
                    f.write(f"Window Title: {data['window_info'].get('title', 'N/A')}\\n")
                    f.write("\\n=== Extracted Text ===\\n")
                    f.write(data.get('screen_text', 'No text extracted'))
            
            messagebox.showinfo("Export Complete", f"Successfully exported {len(self.captured_data)} captures to:\\n{export_folder}")
            
        except Exception as e:
            messagebox.showerror("Export Failed", f"Failed to export data: {str(e)}")

if __name__ == "__main__":
    app = ScreenOverlay()
