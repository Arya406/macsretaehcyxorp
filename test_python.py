import sys
print("Python version:")
print(sys.version)
print("\nPython executable:")
print(sys.executable)

# Test basic functionality
try:
    import tkinter as tk
    print("\ntkinter is available")
    root = tk.Tk()
    root.title("Test Window")
    label = tk.Label(root, text="Python is working!")
    label.pack(padx=20, pady=20)
    print("Showing test window...")
    root.after(3000, root.destroy)  # Close after 3 seconds
    root.mainloop()
    print("Test window closed")
except Exception as e:
    print(f"Error: {e}")
