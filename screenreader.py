# Standard library imports
import os
import sys
import time
import json
import random
import re
import openai
import uuid
import queue
import wave
import base64
import io
import threading
import hashlib
import tkinter.scrolledtext as scrolledtext
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
from datetime import datetime, timedelta

# Try to import Windows-specific modules
try:
    import win32gui
    import win32con
    import win32process
    import psutil
    WINDOWS_MODULES_AVAILABLE = True
except ImportError:
    WINDOWS_MODULES_AVAILABLE = False
    print("Warning: Windows modules not available. Some features will be limited.")

# Base window info class
class BaseWindowInfo:
    def __init__(self, hwnd=None, title="Unknown", process_name="Unknown"):
        self.hwnd = hwnd
        self.title = title
        self.process_name = process_name
        self.last_updated = time.time()
    
    def __str__(self):
        return f"{self.process_name} - {self.title}"

# Dummy implementation for when Windows modules aren't available
class DummyWindowInfo(BaseWindowInfo):
    pass

# Real implementation for Windows
if WINDOWS_MODULES_AVAILABLE:
    class WindowInfo(BaseWindowInfo):
        pass
else:
    WindowInfo = DummyWindowInfo
from time import time as current_timestamp
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union, Tuple, Callable

class LLMIntegration:
    """Handles integration with the Language Model for screen context understanding using Groq."""
    
    def __init__(self):
        self.client = None
        self.conversation_history = []
        self.initialized = False
        self._init_client()
    
    def _init_client(self):
        """Initialize the Groq client."""
        try:
            self.client = LLMClient()
            # Test if client is properly initialized
            if hasattr(self.client, 'available_models') and self.client.available_models:
                self.initialized = True
                print("Groq client initialized successfully.")
            else:
                print("Warning: Groq client initialized but no models available.")
        except Exception as e:
            print(f"Warning: Failed to initialize Groq client: {e}")
    
    def query_llm(self, question: str, screen_text: str = "") -> str:
        """
        Query the LLM with a question and include the current screen text as context.
        
        Args:
            question: The question to ask
            screen_text: The current text extracted from the screen
            
        Returns:
            str: The LLM's response or an error message
        """
        if not self.initialized or not self.client:
            return "Error: Groq client not properly initialized. Please check your GROQ_API_KEY and internet connection."
            
        try:
            # Create a prompt that includes the screen context
            prompt = f"""You are a helpful assistant that helps users understand and interact with their screen content.
            
Current screen content:
```
{screen_text}
```

User's question: {question}

Please provide a helpful response based on the screen content above. If the question requires information not present on screen, please state that clearly."""
            
            # Use the first available model (llama3-8b-8192 by default)
            model = self.client.available_models[0] if hasattr(self.client, 'available_models') and self.client.available_models else "llama3-8b-8192"
            
            # Get response from Groq
            response = self.client.get_response(prompt, model=model)
            
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
            
        except Exception as e:
            return f"Error querying LLM: {str(e)}"

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
from PIL import Image, ImageTk, ImageGrab, ImageEnhance, ImageFilter, ImageDraw, ImageFont
from textblob import TextBlob

# Initialize Groq client (set your API key in environment variables)
class LLMClient:
    def __init__(self, api_key: Optional[str] = None):
        from groq import Groq
        
        # Try to get API key from environment or parameter
        api_key = api_key or os.getenv('GROQ_API_KEY')
        
        # If still no key, prompt the user
        if not api_key:
            try:
                import tkinter as tk
                from tkinter import simpledialog
                root = tk.Tk()
                root.withdraw()  # Hide the root window
                api_key = simpledialog.askstring("Groq API Key", 
                                               "Please enter your Groq API key:", 
                                               show='*')
                root.destroy()
                
                if not api_key:
                    raise ValueError("No API key provided")
                    
                # Save to environment for future use
                os.environ['GROQ_API_KEY'] = api_key
            except Exception as e:
                print("Error getting API key:", str(e))
                raise ValueError("API key is required. Please set GROQ_API_KEY environment variable or provide it when initializing LLMClient.")
        
        self.client = Groq(api_key=api_key)
        self.available_models = [
            "llama3-8b-8192",
            "llama3-70b-8192",
            "mixtral-8x7b-32768"
        ]
    
    def get_response(self, prompt: str, model: str = None) -> str:
        try:
            # Use the first available model if none specified
            model_to_use = model or self.available_models[0]
            
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_to_use,
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stream=False,
                stop=None,
            )
            return response.choices[0].message.content
        except Exception as e:
            # Try next available model if one fails
            if not model and len(self.available_models) > 1:
                print(f"Error with model {model_to_use}, trying next available model...")
                self.available_models.pop(0)
                return self.get_response(prompt)
            return f"Error getting LLM response: {str(e)}"

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
    """Main application class for the screen overlay."""
    
    def get_window_info(self, hwnd: int = None) -> Optional[Any]:
        """Get information about a window by its handle."""
        if not WINDOWS_MODULES_AVAILABLE:
            return DummyWindowInfo()
            
        try:
            if hwnd is None or not win32gui.IsWindow(hwnd):
                return None
                
            # Get window title
            title = win32gui.GetWindowText(hwnd)
            if not title:
                return None
                
            # Get process name
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            process = psutil.Process(pid)
            process_name = process.name()
            
            return WindowInfo(hwnd, title, process_name)
            
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError) as e:
            print(f"Error getting window info: {e}")
            return DummyWindowInfo()
    
    def get_foreground_window_info(self) -> Any:
        """Get information about the currently active window."""
        if not WINDOWS_MODULES_AVAILABLE:
            return DummyWindowInfo()
            
        try:
            hwnd = win32gui.GetForegroundWindow()
            return self.get_window_info(hwnd) or DummyWindowInfo()
        except Exception as e:
            print(f"Error getting foreground window: {e}")
            return DummyWindowInfo()
    
    def on_close(self, event=None):
        """Handle window close event and clean up resources."""
        print("Shutting down application...")
        
        # Set flags to stop all running threads
        self.running = False
        self.is_capturing = False
        self.recording = False
        
        # Stop any ongoing recordings
        if hasattr(self, 'recording_thread') and self.recording_thread and self.recording_thread.is_alive():
            print("Stopping audio recording thread...")
            self.recording_thread.join(timeout=2.0)
        
        # Stop any ongoing audio streams
        if hasattr(self, 'stream') and self.stream:
            print("Stopping audio stream...")
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"Error stopping audio stream: {e}")
    def __init__(self, root=None):
        print("Initializing ScreenOverlay...")
        print("Root parameter:", "Provided" if root is not None else "Not provided")
        
        # Initialize context tracking
        self.context_history = []  # Store previous context for better continuity
        self.last_llm_context_update = 0
        self.llm_context = ""  # Store the current LLM context
        
        # Initialize instance variables first
        self.running = True
        self.is_capturing = False
        self.recording = False
        self._after_ids = []
        self.temp_files = []
        self.context_history = []
        self.last_ocr_text = ""
        self.transcription = ""
        self.notification_label = None
        self.notification_timer = None
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.sample_rate = 44100
        self.channels = 2
        self.system_audio = False
        self.recording_thread = None
        self.last_capture_time = 0
        self.capture_interval = 1.0
        self.current_region = None
        self.ocr_cache = {}
        self.min_text_change_threshold = 0.8
        self.llm_client = None
        self.llm_overlay = None
        self.canvas = None
        self.captured_data = []  # Initialize captured_data
        self.root = None  # Initialize root to None initially
        
        # Preview window components
        self.preview_window = None
        self.preview_text = None
        self.last_extracted_text = ""
        
        # Live capture variables
        self.live_capture_active = False
        self.capture_thread = None
        self.stop_capture = threading.Event()
        self.capture_frame = None
        self.capture_delay = 1.0  # seconds between captures
        
        # LLM and context tracking
        try:
            self.llm = LLMIntegration()
            if not self.llm.client:
                print("Warning: Failed to initialize Groq client. LLM features will be disabled.")
        except Exception as e:
            print(f"Warning: Failed to initialize LLM: {e}")
            self.llm = None
            
        self.current_window = None
        self.window_change_callback = None
        self.window_check_interval = 1.0  # seconds
        self.window_check_thread = None
        self.last_window_check = 0
        self.last_screenshot = None
        self.last_extracted_text = ""
        self.context_update_callbacks = []
        
        try:
            # Initialize main window
            if root is None:
                self.root = tk.Tk()
            else:
                self.root = root
            
            # Get screen dimensions for reference
            temp_root = tk.Tk()
            screen_width = temp_root.winfo_screenwidth()
            screen_height = temp_root.winfo_screenheight()
            temp_root.destroy()
            
            # Set window size to be 1/4 of the screen size
            self.window_width = screen_width // 2
            self.window_height = screen_height // 2
            
            # Calculate position to center the window
            x_position = (screen_width - self.window_width) // 2
            y_position = (screen_height - self.window_height) // 2
            
            # Set window attributes
            self.root.title("Screen Data Overlay")
            self.root.attributes('-alpha', 0.9)
            self.root.attributes('-topmost', True)
            self.root.overrideredirect(False)  # Allow normal window controls
            self.root.geometry(f"{self.window_width}x{self.window_height}+{x_position}+{y_position}")
            self.root.attributes("-transparentcolor", "white")
            self.root.config(bg='white')
            
            # Create a canvas that fills the window
            self.canvas = tk.Canvas(self.root, bg='white', highlightthickness=0)
            self.canvas.pack(fill='both', expand=True)
            
            # Initialize LLM client
            try:
                self.llm_client = LLMClient()
            except Exception as e:
                print(f"Warning: Failed to initialize LLM client: {e}")
                self.llm_client = None
            
            # Set up window close handler
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
            
            # Initialize UI elements
            self.setup_ui()
            
            # Show the window after everything is set up
            print("Showing main window...")
            self.root.deiconify()
            
            # Start the update loop
            self.update()
            
            print("ScreenOverlay initialization complete.")
            print("Window state:", "visible" if self.root.winfo_viewable() else "not visible")
            
        except Exception as e:
            print(f"Error during initialization: {e}")
            if hasattr(self, 'root') and self.root is not None and self.root.winfo_exists():
                self.root.destroy()
            raise
            
    def on_close(self, event=None):
        """Handle window close event and clean up resources."""
        print("Shutting down application...")
        self.running = False
        
        # Cancel any pending after events
        for after_id in self._after_ids:
            try:
                self.root.after_cancel(after_id)
            except:
                pass
        
        # Clean up temporary files
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Error removing temporary file {temp_file}: {e}")
        
        # Stop any running threads
        if hasattr(self, 'recording_thread') and self.recording_thread is not None:
            self.recording = False
            if self.recording_thread.is_alive():
                self.recording_thread.join(timeout=2.0)
        
        # Close LLM overlay if open
        if hasattr(self, 'llm_overlay') and self.llm_overlay:
            print("Closing LLM overlay...")
            try:
                self.llm_overlay.destroy()
            except Exception as e:
                print(f"Error closing LLM overlay: {e}")
        os._exit(0)

    def setup_ui(self):
        """Set up the user interface."""
        # Create a draggable control panel
        self.control_panel = tk.Frame(self.root, bg='#f0f0f0', bd=2, relief='raised')
        self.control_panel.place(x=20, y=20, width=300, height=500)  # Increased height for additional controls
        
        # Make the control panel draggable
        def start_drag(event):
            self.control_panel.startX = event.x
            self.control_panel.startY = event.y
            
        def stop_drag(event):
            self.control_panel.startX = None
            self.control_panel.startY = None
            
        def do_drag(event):
            if hasattr(self.control_panel, 'startX') and self.control_panel.startX is not None:
                dx = event.x - self.control_panel.startX
                dy = event.y - self.control_panel.startY
                x = self.control_panel.winfo_x() + dx
                y = self.control_panel.winfo_y() + dy
                self.control_panel.place(x=x, y=y)
        
        # Bind drag events to the title bar
        self.control_panel.bind('<Button-1>', start_drag)
        self.control_panel.bind('<B1-Motion>', do_drag)
        self.control_panel.bind('<ButtonRelease-1>', stop_drag)
        
        # Add Preview button to control panel
        self.preview_btn = ttk.Button(
            self.control_panel, 
            text="Show Extracted Text", 
            command=self.toggle_preview_window,
            width=20
        )
        self.preview_btn.pack(fill='x', padx=5, pady=5)
        
        # Add Process with LLM button
        self.process_llm_btn = ttk.Button(
            self.control_panel,
            text="Process with LLM",
            command=self.process_with_llm,
            width=20
        )
        self.process_llm_btn.pack(fill='x', padx=5, pady=5)
        
        # Add Live Capture button
        self.live_capture_btn = ttk.Button(
            self.control_panel,
            text="Start Live Capture",
            command=self.toggle_live_capture,
            width=20
        )
        self.live_capture_btn.pack(fill='x', padx=5, pady=5)
        
        # Add capture delay slider
        delay_frame = ttk.Frame(self.control_panel)
        delay_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(delay_frame, text="Capture Delay (s):").pack(side='left')
        
        self.delay_var = tk.DoubleVar(value=self.capture_delay)
        self.delay_slider = ttk.Scale(
            delay_frame,
            from_=0.2,
            to=5.0,
            orient='horizontal',
            variable=self.delay_var,
            command=lambda e: setattr(self, 'capture_delay', float(self.delay_var.get()))
        )
        self.delay_slider.pack(side='left', fill='x', expand=True, padx=5)
        
        # Add a label to show current delay
        self.delay_label = ttk.Label(delay_frame, text=f"{self.capture_delay:.1f}s")
        self.delay_label.pack(side='right')
        
        # Bind the slider to update the label
        self.delay_var.trace_add('write', lambda *_: self.delay_label.config(
            text=f"{float(self.delay_var.get()):.1f}s"))
            
        # Add LLM question input
        self.llm_frame = ttk.LabelFrame(self.control_panel, text="Ask about current screen")
        self.llm_frame.pack(fill='x', padx=5, pady=5)
        
        self.llm_question = ttk.Entry(self.llm_frame)
        self.llm_question.pack(fill='x', padx=5, pady=5)
        self.llm_question.bind('<Return>', self._on_ask_question)
        
        self.ask_btn = ttk.Button(
            self.llm_frame,
            text="Ask",
            command=self._on_ask_question
        )
        self.ask_btn.pack(fill='x', padx=5, pady=5)
        
        # Response area
        self.llm_response = scrolledtext.ScrolledText(
            self.llm_frame,
            height=8,
            wrap=tk.WORD,
            state='disabled'
        )
        self.llm_response.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(
            self.llm_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(fill='x', padx=2, pady=2)
        self.status_var.set("Ready")
        
        # Register context update callback
        self.context_update_callbacks.append(self._update_context_ui)
        
        # Add LLM button to control panel
        self.llm_button = ttk.Button(
            self.control_panel, 
            text="Ask LLM", 
            command=self.toggle_llm_overlay,
            width=20
        )
        self.llm_button.pack(fill='x', padx=5, pady=5)
        
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
            print("Capture started")  # Debug print
        else:
            self.capture_btn.config(text="Start Capture")
            self.status_var.set("Status: Paused")
            print("Capture paused")  # Debug print
    
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
            # Update image preview if screenshot is provided
            if screenshot:
                # Create the preview_image_label if it doesn't exist
                if not hasattr(self, 'preview_image_label') or not hasattr(self, 'preview_text'):
                    self._create_preview_components()
                
                # Resize the image to fit in the preview panel
                max_width = 400
                width, height = screenshot.size
                ratio = min(max_width / width, 1.0)  # Don't scale up small images
                new_size = (int(width * ratio), int(height * ratio))
                img = screenshot.resize(new_size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                if hasattr(self, 'preview_image_label'):
                    self.preview_image_label.configure(image=photo)
                    self.preview_image_label.image = photo  # Keep a reference
            
            # Update text preview if preview_text exists
            if hasattr(self, 'preview_text'):
                self.preview_text.config(state='normal')
                self.preview_text.delete('1.0', tk.END)
                self.preview_text.insert(tk.END, text)
                self.preview_text.config(state='disabled')
                
        except Exception as e:
            print(f"Error updating preview: {e}")
    
    def _create_preview_components(self):
        """Create the preview components if they don't exist."""
        try:
            # Create a frame for the preview if it doesn't exist
            if not hasattr(self, 'preview_frame'):
                self.preview_frame = ttk.Frame(self.root)
                self.preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Create image preview label if it doesn't exist
            if not hasattr(self, 'preview_image_label'):
                self.preview_image_label = ttk.Label(self.preview_frame)
                self.preview_image_label.pack(pady=10)
            
            # Create text preview if it doesn't exist
            if not hasattr(self, 'preview_text'):
                self.preview_text = scrolledtext.ScrolledText(
                    self.preview_frame, 
                    wrap=tk.WORD, 
                    width=60, 
                    height=10,
                    state='disabled'
                )
                self.preview_text.pack(fill=tk.BOTH, expand=True, pady=10)
                
        except Exception as e:
            print(f"Error creating preview components: {e}")
    
    def is_similar_text(self, text1, text2, threshold=0.8):
        """Check if two texts are similar above a threshold."""
        if not text1 or not text2:
            return False
            
        # Simple character-based similarity
        set1 = set(text1.split())
        set2 = set(text2.split())
        
        if not set1 and not set2:
            return True
            
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        similarity = intersection / union if union > 0 else 0
        
        return similarity >= threshold
    
    def capture_screen_data(self):
        if not self.is_capturing:
            print("Capture not active, skipping...")
            return
            
        print("\n=== Starting screen capture ===")
        print(f"Current time: {time.ctime()}")
        
        # Initialize variables to avoid UnboundLocalError
        screen_text = ""
        img = None
        x, y = 0, 0
        window_info = None
        
        # Update last capture time for rate limiting UI updates
        self.last_capture_time = time.time()
        
        try:
            # Get active window info (simplified for now)
            try:
                active_window = pyautogui.getActiveWindow()
                window_info = {
                    'title': active_window.title if active_window else 'Unknown',
                    'position': (active_window.left, active_window.top) if active_window else (0, 0),
                    'size': (active_window.width, active_window.height) if active_window else (0, 0)
                }
                print(f"Active window: {window_info['title']}")
            except Exception as e:
                print(f"Error getting active window: {e}")
                window_info = None
            
            # Get mouse position
            try:
                x, y = pyautogui.position()
                print(f"Mouse position: x={x}, y={y}")
                
                # Capture only a region around the mouse (faster than full screen)
                region_size = 800  # pixels
                monitor = {
                    'left': max(0, x - region_size // 2),
                    'top': max(0, y - region_size // 2),
                    'width': min(region_size, pyautogui.size().width - x + region_size // 2),
                    'height': min(region_size, pyautogui.size().height - y + region_size // 2)
                }
                print(f"Capture region: {monitor}")
                
                # Extract text using OCR
                print("\n--- Starting Text Extraction ---")
                try:
                    # Debug: Print available monitors
                    with mss.mss() as sct:
                        print(f"Available monitors: {sct.monitors}")
                        
                        # Capture the screenshot
                        screenshot = sct.grab(monitor)
                        print(f"Screenshot captured: {screenshot.width}x{screenshot.height} pixels")
                        
                        # Convert to PIL Image
                        img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
                        print(f"Image created: {img.size[0]}x{img.size[1]} pixels, mode: {img.mode}")
                        
                        # Create a copy of the image for OCR
                        ocr_img = img.copy()
                        
                        # Try to get text from OCR cache first
                        img_hash = hashlib.md5(ocr_img.tobytes()).hexdigest()
                        if img_hash in self.ocr_cache:
                            screen_text = self.ocr_cache[img_hash]
                            print("Using cached OCR result")
                        else:
                            print("Performing new OCR...")
                            # If not in cache, perform OCR
                            screen_text = self.extract_text_from_image(ocr_img, fast_mode=True)
                            
                            # Cache the result if we got text
                            if screen_text.strip():
                                print("Caching OCR result")
                                self.ocr_cache[img_hash] = screen_text
                                
                                # Clear old cache entries to prevent memory issues
                                if len(self.ocr_cache) > 100:
                                    print("Cleaning up OCR cache...")
                                    # Remove the oldest entries
                                    for _ in range(min(50, len(self.ocr_cache))):
                                        self.ocr_cache.popitem(last=False)
                                    print(f"OCR cache size after cleanup: {len(self.ocr_cache)}")
                        
                        print(f"Final extracted text: {screen_text[:200]}..." if screen_text else "No text extracted")
                        
                        # Store the captured data
                        capture_time = time.time()
                        capture_data = {
                            'timestamp': capture_time,
                            'screenshot': img,
                            'text': screen_text,
                            'mouse_position': (x, y),
                            'window_info': window_info if window_info is not None else None,
                            'full_screen': False
                        }
                        
                        # Process all captures, even if no text is detected
                        self.captured_data.append(capture_data)
                        self.last_ocr_text = screen_text
                        
                        print(f"Capture {len(self.captured_data)} completed. Text length: {len(screen_text)} characters")
                        
                        # Only analyze if we have text
                        if screen_text.strip():
                            # Analyze the captured text for questions or problems
                            self.analyze_and_respond(screen_text, capture_data)
                            
                            # Show a notification of the captured text
                            preview_text = (screen_text[:47] + '...') if len(screen_text) > 50 else screen_text
                            status_msg = f"Captured: {preview_text}"
                            
                            if hasattr(self, 'status_var'):
                                self.status_var.set(status_msg)
                            
                            # Add to log
                            time_str = datetime.fromtimestamp(capture_time).strftime('%H:%M:%S')
                            print(f"[{time_str}] {status_msg}")
                        
                        # Always update preview if possible
                        if hasattr(self, 'update_preview'):
                            self.update_preview(img, screen_text)
                        
                        # Update UI if canvas exists
                        if hasattr(self, 'canvas') and hasattr(self, 'root') and self.root.winfo_exists():
                            try:
                                self.canvas.delete('marker')
                                self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='red', tags='marker')
                                self.canvas.xview_moveto(max(0, x / self.root.winfo_screenwidth() - 0.1))
                                self.canvas.yview_moveto(max(0, y / self.root.winfo_screenheight() - 0.1))
                            except Exception as canvas_error:
                                print(f"Error updating canvas: {canvas_error}")
                
                except Exception as e:
                    print(f"Error during screenshot capture or processing: {e}")
                    import traceback
                    traceback.print_exc()
                    return
                    
            except Exception as e:
                print(f"Error getting mouse position or calculating capture region: {e}")
                return
                
        except Exception as e:
            error_msg = f"Error in capture_screen_data: {str(e)}"
            print(error_msg)
            if hasattr(self, 'status_var') and hasattr(self.status_var, 'set'):
                try:
                    self.status_var.set(f"Error: {str(e)[:100]}")
                except:
                    pass
            
            # Create a copy of the image for OCR
            ocr_img = img.copy()
            
            # Try to get text from OCR cache first
            img_hash = hashlib.md5(ocr_img.tobytes()).hexdigest()
            if img_hash in self.ocr_cache:
                screen_text = self.ocr_cache[img_hash]
                print("Using cached OCR result")
            else:
                print("Performing new OCR...")
                # If not in cache, perform OCR
                screen_text = self.extract_text_from_image(ocr_img, fast_mode=True)
                
                # Cache the result if we got text
                if screen_text.strip():
                    print("Caching OCR result")
                    self.ocr_cache[img_hash] = screen_text
                    
                    # Clear old cache entries to prevent memory issues
                    if len(self.ocr_cache) > 100:
                        print("Cleaning up OCR cache...")
                        # Remove the oldest entries
                        for _ in range(min(50, len(self.ocr_cache))):
                            self.ocr_cache.popitem(last=False)
                        print(f"OCR cache size after cleanup: {len(self.ocr_cache)}")
            
            print(f"Final extracted text: {screen_text[:200]}..." if screen_text else "No text extracted")
            
            # Store the captured data
            capture_time = time.time()
            capture_data = {
                'timestamp': capture_time,
                'screenshot': img,
                'text': screen_text,
                'mouse_position': (x, y),
                'window_info': window_info if window_info is not None else None,
                'full_screen': False
            }
            
            # Process all captures, even if no text is detected
            self.captured_data.append(capture_data)
            self.last_ocr_text = screen_text
            
            # Update LLM context with new text if available
            if screen_text and screen_text.strip():
                self.update_llm_context(screen_text)
            
            print(f"Capture {len(self.captured_data)} completed. Text length: {len(screen_text)} characters")
            
            # Only analyze if we have text
            if screen_text.strip():
                # Analyze the captured text for questions or problems
                self.analyze_and_respond(screen_text, capture_data)
                
                # Show a notification of the captured text
                preview_text = (screen_text[:47] + '...') if len(screen_text) > 50 else screen_text
                status_msg = f"Captured: {preview_text}"
                
                if hasattr(self, 'status_var'):
                    self.status_var.set(status_msg)
                
                # Add to log
                time_str = datetime.fromtimestamp(capture_time).strftime('%H:%M:%S')
                print(f"[{time_str}] {status_msg}")
            
            # Always update preview if possible
            if hasattr(self, 'update_preview'):
                self.update_preview(img, screen_text)
            
            # Update UI if canvas exists
            if hasattr(self, 'canvas') and hasattr(self, 'root') and self.root.winfo_exists():
                try:
                    self.canvas.delete('marker')
                    self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='red', tags='marker')
                    self.canvas.xview_moveto(max(0, x / self.root.winfo_screenwidth() - 0.1))
                    self.canvas.yview_moveto(max(0, y / self.root.winfo_screenheight() - 0.1))
                except Exception as canvas_error:
                    print(f"Error updating canvas: {canvas_error}")
        
        except Exception as e:
            error_msg = f"Error in capture_screen_data: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            if hasattr(self, 'status_var') and hasattr(self.status_var, 'set'):
                try:
                    self.status_var.set(f"Error: {str(e)[:100]}")
                except:
                    pass
    
    def analyze_and_respond(self, text: str, capture_data: dict):
        """Analyze the captured text and generate a response if a question is detected."""
        # Check if the text contains question marks or question words
        question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which', 'can you', 'could you', 'would you']
        has_question_mark = '?' in text
        has_question_word = any(word in text.lower() for word in question_words)
        
        # If it looks like a question, generate a response
        if has_question_mark or has_question_word:
            try:
                # Create a prompt that includes the captured text
                prompt = f"""You are a helpful assistant. Answer the following question concisely and accurately.
                If the text is not a question, just say 'Not a question'.
                
                Text: {text}
                
                Answer:"""
                
                # Get response from LLM
                response = self.llm_client.get_response(prompt)
                
                # Only show the response if it's not the default 'not a question' response
                if response.lower().strip() != 'not a question':
                    # Show the response in the LLM overlay if it's open
                    if hasattr(self, 'llm_overlay') and self.llm_overlay:
                        self.add_message("Assistant", f"I found a question in the captured text:\n\n{text}\n\n{response}")
                    
                    # Also show a notification
                    self.show_notification("Generated response to question", duration=5000)
                    
                    # Save the response with the capture data
                    capture_data['llm_response'] = response
                    
            except Exception as e:
                print(f"Error generating response: {e}")
    
    def toggle_preview_window(self):
        """Toggle the preview window visibility."""
        if hasattr(self, 'preview_window') and self.preview_window and self.preview_window.winfo_exists():
            self.preview_window.destroy()
            self.preview_window = None
        else:
            self.create_preview_window()
    
    def create_preview_window(self):
        """Create the preview window for extracted text."""
        if hasattr(self, 'preview_window') and self.preview_window and self.preview_window.winfo_exists():
            self.preview_window.lift()
            self.preview_window.focus_force()
            return
            
        self.preview_window = tk.Toplevel(self.root)
        self.preview_window.title("Extracted Text Preview")
        self.preview_window.geometry("600x400+100+100")
        self.preview_window.protocol("WM_DELETE_WINDOW", self.toggle_preview_window)
        
        # Add a frame for buttons
        btn_frame = ttk.Frame(self.preview_window)
        btn_frame.pack(fill='x', padx=5, pady=5)
        
        # Add copy button
        copy_btn = ttk.Button(
            btn_frame,
            text="Copy to Clipboard",
            command=self.copy_extracted_text
        )
        copy_btn.pack(side='left', padx=5)
        
        # Add clear button
        clear_btn = ttk.Button(
            btn_frame,
            text="Clear Text",
            command=self.clear_extracted_text
        )
        clear_btn.pack(side='left', padx=5)
        
        # Add text widget with scrollbar
        text_frame = ttk.Frame(self.preview_window)
        text_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.preview_text = tk.Text(
            text_frame,
            wrap='word',
            yscrollcommand=scrollbar.set,
            font=('Consolas', 10),
            padx=5,
            pady=5
        )
        self.preview_text.pack(fill='both', expand=True)
        scrollbar.config(command=self.preview_text.yview)
        
        # Add any previously extracted text
        if hasattr(self, 'last_extracted_text') and self.last_extracted_text:
            self.preview_text.insert('1.0', self.last_extracted_text)
    
    def copy_extracted_text(self):
        """Copy the extracted text to clipboard."""
        if hasattr(self, 'preview_text') and self.preview_text:
            text = self.preview_text.get('1.0', 'end-1c')
            if text:
                self.root.clipboard_clear()
                self.root.clipboard_append(text)
                self.show_notification("Text copied to clipboard!")
    
    def clear_extracted_text(self):
        """Clear the extracted text."""
        if hasattr(self, 'preview_text') and self.preview_text:
            self.preview_text.delete('1.0', 'end')
            self.last_extracted_text = ""
            self.show_notification("Text cleared!")
    
    def update_extracted_text(self, text, source_image=None):
        """Update the preview window with extracted text."""
        if not text and not source_image:
            return
            
        self.last_extracted_text = text or ""
        
        # If preview window is open, update it
        if hasattr(self, 'preview_text') and self.preview_text:
            self.preview_text.config(state='normal')
            self.preview_text.delete('1.0', 'end')
            self.preview_text.insert('end', f"=== Extracted Text ===\n\n{text or 'No text found'}\n\n")
            
            if source_image is not None:
                # Convert the image to PhotoImage and display it
                try:
                    # Resize image to fit in the preview
                    max_width = 500
                    width, height = source_image.size
                    if width > max_width:
                        ratio = max_width / width
                        new_height = int(height * ratio)
                        source_image = source_image.resize((max_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Convert to PhotoImage
                    photo = ImageTk.PhotoImage(source_image)
                    
                    # Create or update image label
                    if not hasattr(self, 'preview_image_label'):
                        self.preview_image_label = ttk.Label(self.preview_text, image=photo)
                        self.preview_image_label.image = photo  # Keep a reference!
                        self.preview_text.window_create('end', window=self.preview_image_label)
                        self.preview_text.insert('end', '\n\n=== Source Image ===\n\n')
                    else:
                        self.preview_image_label.config(image=photo)
                        self.preview_image_label.image = photo  # Keep a reference!
                except Exception as e:
                    print(f"Error displaying image: {e}")
            
            self.preview_text.see('end')
            self.preview_text.config(state='disabled')
        
        # If preview window is closed but we have text, show a notification
        elif text and not (hasattr(self, 'preview_window') and self.preview_window and self.preview_window.winfo_exists()):
            self.show_notification("Text extracted! Click 'Show Extracted Text' to view.")
    
    def toggle_live_capture(self):
        """Toggle live text capture from screen."""
        if self.live_capture_active:
            self.stop_live_capture()
        else:
            self.start_live_capture()
    
    def start_live_capture(self):
        """Start live text capture from screen."""
        if self.live_capture_active:
            return
            
        self.live_capture_active = True
        self.stop_capture.clear()
        self.live_capture_btn.config(text="Stop Live Capture")
        
        # Ensure preview window is open
        if not (hasattr(self, 'preview_window') and self.preview_window and self.preview_window.winfo_exists()):
            self.create_preview_window()
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._live_capture_loop, daemon=True)
        self.capture_thread.start()
        
        self.show_notification("Live capture started!")
    
    def stop_live_capture(self):
        """Stop live text capture."""
        if not self.live_capture_active:
            return
            
        self.live_capture_active = False
        self.stop_capture.set()
        self.live_capture_btn.config(text="Start Live Capture")
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        self.show_notification("Live capture stopped.")
    
    def _update_context_ui(self, window_info, extracted_text):
        """Update the UI when the context changes."""
        if not hasattr(self, 'status_var'):
            return
            
        if window_info:
            self.status_var.set(f"Active: {window_info}")
    


    def update_llm_context(self, text):
        """Update the LLM context with new text if it's different enough from the last update."""
        if not text or not text.strip():
            return
            
        current_time = time.time()
        
        # Only update context if enough time has passed or text is significantly different
        if (current_time - self.last_llm_context_update > 30 or  # 30 seconds have passed
            not self.llm_context or  # No context yet
            len(set(text.split()) - set(self.llm_context.split())) / len(set(text.split())) > 0.3):  # 30% new content
            
            # Keep context history manageable
            if len(self.context_history) >= 5:  # Keep last 5 contexts
                self.context_history.pop(0)
                
            self.context_history.append(text)
            self.llm_context = "\n\n".join(self.context_history)
            self.last_llm_context_update = current_time
    
    def query_llm_with_context(self, question=None):
        """Query the LLM with the current context and optional question."""
        if not self.llm or not self.llm.initialized:
            return "LLM not available"
            
        if not self.llm_context:
            return "No context available. Please capture some text first."
            
        prompt = """You are a helpful assistant that helps users understand and interact with their screen content.
        
        Current screen content (for context):
        ```
        {context}
        ```"""
        
        if question:
            prompt += f"\n\nUser's question: {question}"
        else:
            prompt += "\n\nPlease analyze the above content and provide a summary of the most important information."
            
        prompt += "\n\nPlease provide a helpful response based on the screen content above. If the question requires information not present on screen, please state that clearly."
        
        return self.llm.query_llm(prompt.format(context=self.llm_context))
    
    def process_with_llm(self, question=None):
        """Process the extracted text with LLM and display the response."""
        # Update context with latest text if available
        if hasattr(self, 'last_ocr_text') and self.last_ocr_text.strip():
            self.update_llm_context(self.last_ocr_text)
            
        # If no context is available yet
        if not self.llm_context:
            self.show_notification("No text available to process. Capture some text first.", duration=3000)
            return
            
        # Disable the button while processing
        if hasattr(self, 'process_llm_btn'):
            self.process_llm_btn.config(state='disabled')
        self.status_var.set("Processing with LLM...")
        self.root.update()
        
        def process():
            try:
                response = self.query_llm_with_context(question)
                
                # Display the response in a message box
                self.root.after(0, lambda: messagebox.showinfo(
                    "LLM Analysis", 
                    f"{question if question else 'Analysis'}:\n\n{response}",
                    parent=self.root
                ))
                
                self.root.after(0, self.status_var.set, "Analysis complete")
                
            except Exception as e:
                error_msg = f"Error processing with LLM: {str(e)}"
                print(error_msg)
                self.root.after(0, self.status_var.set, error_msg)
                self.root.after(0, lambda: messagebox.showerror("Error", error_msg, parent=self.root))
            finally:
                # Re-enable the button
                if hasattr(self, 'process_llm_btn'):
                    self.root.after(0, lambda: self.process_llm_btn.config(state='normal'))
        
        # Start processing in a separate thread
        threading.Thread(target=process, daemon=True).start()
        
    def _on_ask_question(self, event=None):
        """Handle asking a question about the current screen."""
        question = self.llm_question.get().strip()
        if not question:
            return
            
        # Get the most recent screen text if available
        screen_text = getattr(self, 'last_ocr_text', '')
        
        # Disable input while processing
        self.llm_question.config(state='disabled')
        self.ask_btn.config(state='disabled')
        self.status_var.set("Thinking...")
        self.root.update()
        
        # Process in a separate thread to keep UI responsive
        def process_question():
            try:
                if not self.llm or not self.llm.initialized:
                    self.root.after(0, self.status_var.set, "LLM not available")
                    return
                    
                # Pass both the question and the current screen text
                response = self.llm.query_llm(question, screen_text)
                self.root.after(0, self.status_var.set, f"LLM: {response}")
                
                # Add to chat history if available
                if hasattr(self, 'chat_history') and response:
                    self.chat_history.insert(tk.END, f"You: {question}\n")
                    self.chat_history.insert(tk.END, f"AI: {response}\n\n")
                    
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                self.root.after(0, self.status_var.set, error_msg)
                print(error_msg)
            finally:
                # Re-enable input and clear question
                self.root.after(0, lambda: [
                    self.llm_question.config(state='normal'),
                    self.ask_btn.config(state='normal'),
                    self.llm_question.delete(0, tk.END),
                    self.llm_question.focus()
                ])
        
        # Start the processing thread
        threading.Thread(target=process_question, daemon=True).start()
    
    def _display_response(self, question, response):
        """Display the question and response in the chat area."""
        self.llm_response.config(state='normal')
        
        # Add question
        self.llm_response.insert('end', f"You: {question}\n\n")
        
        # Add response
        self.llm_response.insert('end', f"Assistant: {response}\n")
        self.llm_response.insert('end', "-" * 50 + "\n\n")
        
        # Auto-scroll to bottom
        self.llm_response.see('end')
        self.llm_response.config(state='disabled')
        
        # Clear the question field
        self.llm_question.delete(0, 'end')
        self.status_var.set("Ready")
    
    def _live_capture_loop(self):
        """Main loop for live text capture."""
        last_capture_time = 0
        
        while self.live_capture_active and not self.stop_capture.is_set():
            try:
                current_time = time.time()
                
                # Only capture if enough time has passed
                if current_time - last_capture_time >= self.capture_delay:
                    # Capture screen
                    screenshot = ImageGrab.grab()
                    
                    # Extract text
                    text = self.extract_text_from_image(screenshot)
                    
                    # Update UI in main thread
                    self.root.after(0, self.update_extracted_text, text, screenshot)
                    
                    last_capture_time = current_time
                
                # Small sleep to prevent CPU overload
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error in live capture: {e}")
                time.sleep(1)  # Prevent tight loop on error
                
                # Try to recover by stopping and restarting capture
                if self.live_capture_active:
                    self.root.after(0, self.stop_live_capture)
                    break
    
    def extract_text_from_image(self, img, fast_mode=True):
        """
        Extract text from image using Tesseract OCR with advanced preprocessing.
        
        Args:
            img: Input image (PIL Image or OpenCV format)
            fast_mode: If True, uses faster but less accurate processing
            
        Returns:
            str: Extracted and formatted text
        """
        print("\n=== Starting Advanced OCR processing ===")
        try:
            # Verify Tesseract is installed and accessible
            try:
                pytesseract.get_tesseract_version()
            except pytesseract.TesseractNotFoundError:
                error_msg = "ERROR: Tesseract is not installed or not in system PATH"
                print(error_msg)
                return error_msg

            # Start timing the OCR process
            start_time = time.time()
            
            # Create debug directory if it doesn't exist
            debug_dir = "debug_ocr"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            
            # Convert PIL Image to OpenCV format if needed
            if isinstance(img, Image.Image):
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                pil_img = img.convert('RGB')
            else:
                img_cv = img.copy()
                pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

            # Save original image for reference
            timestamp = int(time.time())
            cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_original.png"), img_cv)
            
            if fast_mode:
                print("Using fast OCR mode")
                # Simple preprocessing for fast mode
                img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                _, img_processed = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                config = '--psm 3 --oem 3 -c preserve_interword_spaces=1'
            else:
                print("Using detailed OCR mode")
                # Advanced preprocessing for better accuracy
                
                # Convert to grayscale
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                
                # Noise reduction
                denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
                
                # Increase contrast
                alpha = 1.5  # Contrast control (1.0-3.0)
                beta = 0     # Brightness control (0-100)
                enhanced = cv2.convertScaleAbs(denoised, alpha=alpha, beta=beta)
                
                # Apply Gaussian blur to reduce noise
                blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
                
                # Adaptive thresholding
                img_processed = cv2.adaptiveThreshold(
                    blurred, 255, 
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
                
                # Invert image if needed (black text on white background)
                img_processed = cv2.bitwise_not(img_processed)
                
                # Save processed image for debugging
                cv2.imwrite(os.path.join(debug_dir, f"{timestamp}_processed.png"), img_processed)
                
                # Advanced Tesseract configuration
                config = (
                    '--oem 3 '  # LSTM OCR Engine
                    '--psm 3 '  # Fully automatic page segmentation, but no OSD
                    '-c preserve_interword_spaces=1 '  # Preserve spaces between words
                    r'tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ\'"\-.,:;!?()[]{}<>/\\@#$%^&*_+=|~` '  # Allowed characters
                )
            
            # Add language support (you can add more languages as needed)
            config += '-l eng'
            
            # For debugging, print the exact command that would be used
            print(f"Tesseract config: {config}")
            
            # Perform OCR with the preprocessed image
            print(f"Running Tesseract with config: {config}")
            try:
                text = pytesseract.image_to_string(
                    img_processed,
                    config=config.strip(),
                    timeout=10  # 10 second timeout
                )
                
                # Post-processing for better text quality
                if text:
                    # First, clean up the text
                    text = text.strip()
                    
                    # If we got no text, try with a different PSM
                    if not text and '--psm 3' in config:
                        print("No text found with PSM 3, trying PSM 6")
                        alt_config = config.replace('--psm 3', '--psm 6')
                        text = pytesseract.image_to_string(
                            img_processed,
                            config=alt_config.strip(),
                            timeout=10
                        )
                    
                    # If still no text, try with a simpler configuration
                    if not text and '--oem 3' in config:
                        print("No text found with OEM 3, trying OEM 1")
                        alt_config = config.replace('--oem 3', '--oem 1')
                        text = pytesseract.image_to_string(
                            img_processed,
                            config=alt_config.strip(),
                            timeout=10
                        )
                    
                    # Basic text cleaning
                    if text:
                        # Remove unwanted characters and normalize spaces
                        text = ' '.join(text.split())
                        
                        # Fix common OCR mistakes
                        replacements = {
                            '|': 'I',
                            '1': 'I',
                            '0': 'O',
                            '5': 'S',
                            '2': 'Z',
                            '8': 'B',
                            '\\': '/',
                            '``': '"',
                            "''": '"',
                            ' .': '.',
                            ' ,': ',',
                            ' ;': ';',
                            ' :': ':',
                            ' !': '!',
                            ' ?': '?'
                        }
                        
                        for old, new in replacements.items():
                            text = text.replace(old, new)
                        
                        # Add proper spacing after punctuation if missing
                        text = re.sub(r'(?<=[.!?])(?=[^\s\d])', ' ', text)
                        
                        # Remove any remaining double spaces
                        text = ' '.join(text.split())
                        
                        # Update the preview window with extracted text
                        self.update_extracted_text(text)
                    else:
                        print("Warning: No text could be extracted from the image")
                
                # Calculate processing time
                processing_time = time.time() - start_time
                print(f"OCR processing completed in {processing_time:.2f} seconds")
                
                # Log the extracted text for debugging
                if text.strip():
                    print(f"Extracted text length: {len(text)} characters")
                    print(f"First 200 chars: {text[:200]}..." if len(text) > 200 else f"Extracted text: {text}")
                else:
                    print("No text detected in image")
                
                return text.strip()
                
            except pytesseract.TesseractError as te:
                error_msg = f"Tesseract Error: {str(te)}"
                print(error_msg)
                return error_msg
                
        except Exception as e:
            error_msg = f"Error during OCR processing: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg
    
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
        if not self.running or not hasattr(self, 'root') or not self.root.winfo_exists():
            return
            
        try:
            if self.is_capturing:
                self.capture_screen_data()
            self.root.after(100, self.update)  # Schedule next update
        except Exception as e:
            print(f"Error in update loop: {e}")
            if hasattr(self, 'root') and self.root.winfo_exists():
                self.root.after(100, self.update)
    
    def toggle_llm_overlay(self):
        """Toggle the LLM overlay window."""
        if hasattr(self, 'llm_overlay') and self.llm_overlay and self.llm_overlay.winfo_exists():
            self.llm_overlay.destroy()
            self.llm_overlay = None
        else:
            self.create_llm_overlay()
    
    def create_llm_overlay(self):
        """Create the LLM overlay window."""
        self.llm_overlay = tk.Toplevel(self.root)
        self.llm_overlay.overrideredirect(True)
        self.llm_overlay.attributes('-topmost', True)
        self.llm_overlay.geometry('400x300+500+100')
        self.llm_overlay.configure(bg='#2c3e50')
        
        # Make window draggable
        def start_move(event):
            self.llm_overlay.x = event.x
            self.llm_overlay.y = event.y

        def stop_move(event):
            self.llm_overlay.x = None
            self.llm_overlay.y = None

        def do_move(event):
            deltax = event.x - self.llm_overlay.x
            deltay = event.y - self.llm_overlay.y
            x = self.llm_overlay.winfo_x() + deltax
            y = self.llm_overlay.winfo_y() + deltay
            self.llm_overlay.geometry(f"+{x}+{y}")
            
        # Title bar
        title_bar = tk.Frame(self.llm_overlay, bg='#34495e', relief='raised', bd=1)
        title_bar.pack(fill='x')
        
        title_label = tk.Label(title_bar, text="LLM Assistant", bg='#34495e', fg='white')
        title_label.pack(side='left', padx=5)
        
        close_btn = tk.Label(title_bar, text='×', bg='#34495e', fg='white', cursor='hand2')
        close_btn.pack(side='right', padx=5)
        close_btn.bind('<Button-1>', lambda e: self.llm_overlay.destroy())
        
        title_bar.bind('<Button-1>', start_move)
        title_bar.bind('<B1-Motion>', do_move)
        title_bar.bind('<ButtonRelease-1>', stop_move)
        
        # Input area
        input_frame = tk.Frame(self.llm_overlay, bg='#2c3e50', padx=5, pady=5)
        input_frame.pack(fill='x', side='bottom')
        
        self.llm_input = scrolledtext.ScrolledText(input_frame, height=3, width=40, wrap=tk.WORD)
        self.llm_input.pack(fill='x', pady=(0, 5))
        
        send_btn = ttk.Button(
            input_frame, 
            text="Send", 
            command=self.send_to_llm,
            width=10
        )
        send_btn.pack(side='right')
        
        # Response area
        response_frame = tk.Frame(self.llm_overlay, bg='#2c3e50')
        response_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.llm_response = scrolledtext.ScrolledText(
            response_frame, 
            wrap=tk.WORD, 
            bg='#34495e', 
            fg='white',
            insertbackground='white',
            font=('Arial', 10)
        )
        self.llm_response.pack(fill='both', expand=True)
        self.llm_response.config(state='disabled')
        
        # Bind Enter key to send message (Shift+Enter for new line)
        self.llm_input.bind('<Return>', lambda e: 'break' if not e.state else self.send_to_llm())
        self.llm_input.bind('<Shift-Return>', lambda e: self.llm_input.insert(tk.END, '\n'))
    
    def send_to_llm(self):
        """Send the current input to the LLM and display the response."""
        user_input = self.llm_input.get('1.0', tk.END).strip()
        if not user_input:
            return
            
        # Add user message to chat
        self.add_message("You", user_input)
        self.llm_input.delete('1.0', tk.END)
        
        # Get LLM response (in a separate thread to avoid freezing the UI)
        def get_llm_response():
            try:
                # Add context from the last capture if available
                context = ""
                if hasattr(self, 'last_ocr_text') and self.last_ocr_text:
                    context = f"Context from screen: {self.last_ocr_text[:1000]}\n\n"
                
                response = self.llm_client.get_response(f"{context}Question: {user_input}")
                self.add_message("Assistant", response)
            except Exception as e:
                self.add_message("Error", f"Failed to get LLM response: {str(e)}")
        
        # Start the LLM request in a separate thread
        threading.Thread(target=get_llm_response, daemon=True).start()
    
    def add_message(self, sender: str, message: str):
        """Add a message to the chat display."""
        self.llm_response.config(state='normal')
        self.llm_response.insert(tk.END, f"{sender}: {message}\n\n")
        self.llm_response.see(tk.END)
        self.llm_response.config(state='disabled')
    
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

def main():
    try:
        print("1. Starting main() function...")
        print("2. Creating ScreenOverlay instance...")
        app = ScreenOverlay()
        print("3. ScreenOverlay instance created")
        print("4. Starting main loop...")
        app.root.mainloop()
        print("5. Main loop ended")  # This should not be reached until the app is closed
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        if 'app' in locals():
            try:
                app.on_close()
            except:
                pass
        sys.exit(1)

if __name__ == "__main__":
    print("0. Script started")
    main()
    print("6. Script ended")