import tkinter as tk

print("Starting test GUI...")
root = tk.Tk()
root.title("Test GUI")
label = tk.Label(root, text="If you can see this, the GUI is working!")
label.pack(padx=20, pady=20)
root.mainloop()
