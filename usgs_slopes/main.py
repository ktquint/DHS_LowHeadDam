import tkinter as tk
from tkinter import messagebox, filedialog
from classes import StreamGage


def __main__():
    def browse_folder():
        folder = filedialog.askdirectory()
        if folder:
            folder_var.set(folder)

    def submit():
        gage_id = entry.get()
        folder = folder_var.get()
        if not folder:
            messagebox.showwarning("Missing Folder", "Please select a folder to download the .gpkg file.")
            return
        try:
            gage = StreamGage(gage_id)
            gage.download_nhd(folder)
            gage.merge_metadata()
            messagebox.showinfo("StreamGage Info", f"{gage}\n\nDownload folder: {folder}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to retrieve gage info: {e}")

    root = tk.Tk()
    root.title("USGS StreamGage Info")

    tk.Label(root, text="Enter USGS StreamGage ID:").pack(pady=5)
    entry = tk.Entry(root)
    entry.pack(pady=5)

    folder_frame = tk.Frame(root)
    folder_frame.pack(pady=5)

    folder_var = tk.StringVar()
    folder_entry = tk.Entry(folder_frame, textvariable=folder_var, width=40)
    folder_entry.pack(side=tk.LEFT, padx=5)
    browse_button = tk.Button(folder_frame, text="Browse", command=browse_folder)
    browse_button.pack(side=tk.LEFT)

    tk.Button(root, text="Submit", command=submit).pack(pady=10)

    root.mainloop()

# Run the GUI
if __name__ == '__main__':
    __main__()

