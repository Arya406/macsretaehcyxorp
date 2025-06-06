import soundcard as sc
import time

def list_audio_devices():
    print("\n=== Speakers ===")
    speakers = sc.all_speakers()
    for i, speaker in enumerate(speakers):
        print(f"{i}: {speaker}")
    
    print("\n=== Microphones ===")
    mics = sc.all_microphones(include_loopback=True)
    for i, mic in enumerate(mics):
        print(f"{i}: {mic} (Loopback: {'loopback' in str(mic).lower()})")

if __name__ == "__main__":
    print("Testing soundcard library...")
    list_audio_devices()
