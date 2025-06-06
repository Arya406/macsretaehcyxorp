import sys
import os

def main():
    print("1. Starting test script...")
    print(f"2. Python executable: {sys.executable}")
    print(f"3. Working directory: {os.getcwd()}")
    
    try:
        with open('test_output_file.txt', 'w') as f:
            f.write("Test output at the start of the script\n")
        print("4. Successfully wrote to test_output_file.txt")
    except Exception as e:
        print(f"Error writing to file: {e}")
    
    input("5. Press Enter to exit...")

if __name__ == "__main__":
    main()
