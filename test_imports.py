import sys
print(f"Python version: {sys.version}")
print("\nTesting imports:")

try:
    import pyautogui
    print("✅ pyautogui")
except ImportError as e:
    print(f"❌ pyautogui: {e}")

try:
    import PIL
    print("✅ Pillow")
except ImportError as e:
    print(f"❌ Pillow: {e}")

try:
    import numpy
    print(f"✅ numpy {numpy.__version__}")
except ImportError as e:
    print(f"❌ numpy: {e}")

try:
    import cv2
    print(f"✅ opencv-python {cv2.__version__}")
except ImportError as e:
    print(f"❌ opencv-python: {e}")

try:
    import pytesseract
    print("✅ pytesseract")
except ImportError as e:
    print(f"❌ pytesseract: {e}")

try:
    import mss
    print("✅ mss")
except ImportError as e:
    print(f"❌ mss: {e}")

try:
    import sounddevice
    print("✅ sounddevice")
except ImportError as e:
    print(f"❌ sounddevice: {e}")

try:
    import soundfile
    print("✅ soundfile")
except ImportError as e:
    print(f"❌ soundfile: {e}")

try:
    import speech_recognition
    print("✅ SpeechRecognition")
except ImportError as e:
    print(f"❌ SpeechRecognition: {e}")

try:
    import pyaudio
    print("✅ PyAudio")
except ImportError as e:
    print(f"❌ PyAudio: {e}")

try:
    import soundcard
    print("✅ soundcard")
except ImportError as e:
    print(f"❌ soundcard: {e}")
