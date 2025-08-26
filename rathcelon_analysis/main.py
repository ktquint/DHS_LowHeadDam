import os
import ast
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
import geopandas as gpd
from classes import Dam
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# noinspection PyBroadException,PyTypeChecker
def update_dropdown():
    results_dir = results_entry.get()
    if os.path.isdir(results_dir):
        dam_strs = successful_runs(results_dir)
        dams = sorted([int(d) for d in dam_strs])
        # dams = sorted([int(d) for d in os.listdir(results_dir) if d != '.DS_Store'])
        dams.insert(0, "All Dams")
        dropdown['values'] = dams
        if dams:
            dropdown.set(dams[0])  # select first item
    else:
        dropdown.set("Invalid path")


# this guy selects the csv database
def select_csv_file():
    file_path = filedialog.askopenfilename(
        title="Select a CSV file",
        filetypes=[("CSV files", "*.csv")]
    )
    if file_path:
        database_entry.delete(0, tk.END)
        database_entry.insert(0, file_path)


def select_results_dir():
    results_path = filedialog.askdirectory()
    if results_path:
        results_entry.delete(0, tk.END)
        results_entry.insert(0, results_path)
        update_dropdown()


def plot_shj_estimates():
    lhd_df = pd.read_csv(database_entry.get())
    # Create new window
    win = tk.Toplevel()
    win.title("All Summary Figures")
    win.geometry("1200x600")

    # Create a canvas with scrollbar
    canvas = tk.Canvas(win)
    scrollbar = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
    scroll_frame = ttk.Frame(canvas)

    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    for i in range(1, 5):
        y_t_strs = lhd_df[f'y_t_{i}'].to_list()
        y_t_list = [num for item in y_t_strs for num in ast.literal_eval(item)]

        y_flip_strs = lhd_df[f'y_flip_{i}'].to_list()
        y_flip_list = [num for item in y_flip_strs for num in ast.literal_eval(item)]

        y_2_strs = lhd_df[f'y_2_{i}'].to_list()
        y_2_list = [num for item in y_2_strs for num in ast.literal_eval(item)]

        slopes = lhd_df[f's_{i}'].to_list()
        dam_ids = lhd_df['ID'].tolist()

        nested_list = [ast.literal_eval(item) for item in y_t_strs]
        expanded_slopes = [val for val, group in zip(slopes, nested_list) for _ in range(len(group))]
        expanded_ids = [val for val, group in zip(dam_ids, nested_list) for _ in range(len(group))]


        df = pd.DataFrame({
            'slope': expanded_slopes,
            'y_t': y_t_list,
            'y_flip': y_flip_list,
            'y_2': y_2_list,
            'dam_id': expanded_ids
        })

        df = df.sort_values(['dam_id', 'slope']).reset_index(drop=True)
        x_vals = np.arange(len(df))

        slope = df['slope']
        conjugate = df['y_2'] * 3.281
        flip = df['y_flip'] * 3.281
        tailwater = df['y_t'] * 3.281
        x_labels = slope.round(6).astype(str)

        # Create a Matplotlib figure and embed it
        fig, ax = plt.subplots(figsize=(11, 5))
        cap_width = 0.2

        for x, y, y2, y_flip in zip(x_vals, tailwater, conjugate, flip):
            if y2 < y < y_flip:
                c = 'red'
            elif y >= y_flip:
                c = 'green'
            else:
                c = 'blue'

            ax.vlines(x, y2, y_flip, color='black', linewidth=1)
            ax.hlines(y2, x - cap_width, x + cap_width, color='black', linewidth=1)
            ax.hlines(y_flip, x - cap_width, x + cap_width, color='black', linewidth=1)
            ax.scatter(x, y, color=c, marker='x', zorder=3)

        current_id = None
        start_idx = 0
        shade = True  # start with shading

        for idx, dam in enumerate(df['dam_id']):
            if dam != current_id:
                if current_id is not None and shade:
                    ax.axvspan(start_idx - 0.5, idx - 0.5, color='gray', alpha=0.1)
                current_id = dam
                start_idx = idx
                shade = not shade

        # Handle the final dam
        if shade:
            ax.axvspan(start_idx - 0.5, len(df) - 0.5, color='gray', alpha=0.1)

        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_labels, rotation=90)
        ax.set_xlabel('Slope')
        ax.set_ylabel('Depth (ft)')
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.set_title(f"Summary of Results from Cross-Section No. {i}")
        fig.tight_layout()

        fig_canvas = FigureCanvasTkAgg(fig, master=scroll_frame)
        fig_canvas.draw()
        fig_canvas.get_tk_widget().pack(padx=10, pady=10, fill="both", expand=True)


def plot_cross_section(dam):
    # Create new window
    win = tk.Toplevel()
    win.title("All Cross-Sections")
    win.geometry("800x800")

    # Create a canvas with scrollbar
    canvas = tk.Canvas(win)
    scrollbar = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
    scroll_frame = ttk.Frame(canvas)

    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Generate and add figures
    for cross_section in dam.cross_sections:
        xs_fig = cross_section.plot_cross_section()
        fig_canvas = FigureCanvasTkAgg(xs_fig, master=scroll_frame)
        fig_canvas.draw()
        fig_canvas.get_tk_widget().pack(padx=10, pady=10, fill="both", expand=True)


def plot_rating_curves(dam):
    # Create new window
    win = tk.Toplevel()
    win.title("All Rating Curves")
    win.geometry("800x800")

    # Create a canvas with scrollbar
    canvas = tk.Canvas(win)
    scrollbar = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
    scroll_frame = ttk.Frame(canvas)

    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Generate and add figures
    for cross_section in dam.cross_sections[1:]:
        rc_fig = cross_section.create_combined_fig()
        fig_canvas = FigureCanvasTkAgg(rc_fig, master=scroll_frame)
        fig_canvas.draw()
        fig_canvas.get_tk_widget().pack(padx=10, pady=10, fill="both", expand=True)


def plot_map(dam):
    map_fig = dam.plot_map()
    # Create a new Tkinter window for the plot
    plot_window = tk.Toplevel(root)
    plot_window.title("Dam Map")
    plot_window.geometry("1200x1000")

    # Embed the figure in the Tkinter window
    canvas = FigureCanvasTkAgg(map_fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def plot_wsp(dam):
    wsp_fig = dam.plot_water_surface()
    plot_window = tk.Toplevel(root)
    plot_window.title("Rating Curves")
    plot_window.geometry("1200x1000")
    # Embed the figure in the Tkinter window
    canvas = FigureCanvasTkAgg(wsp_fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def successful_runs(results_dir):
    dam_nos = [str(d) for d in os.listdir(results_dir) if d != '.DS_Store']
    # make a list of all the directories with results
    for i in range(0, len(dam_nos)):
        dam_nos[i] = os.path.join(results_dir, dam_nos[i])

    # make a new list of successful_runs
    successes = []

    for run_results_dir in dam_nos:
        lhd_id = os.path.basename(run_results_dir)
        local_vdt_gpkg = os.path.join(run_results_dir, "VDT", f"{lhd_id}_Local_VDT_Database.gpkg")
        if not os.path.exists(local_vdt_gpkg):
            continue
        try:
            local_vdt_gdf = gpd.read_file(local_vdt_gpkg)
            if not local_vdt_gdf.empty:
                successes.append(lhd_id)
            else:
                continue
        except Exception as e:
            print(f"Error reading DBF for {lhd_id}: {e}")
            return False
    return successes

# Your custom function using the folder path from the entry
def process_ARC():
    # project_dir holds all the project files
    database_csv = database_entry.get()
    results_dir = results_entry.get()

    selected_model = model_var.get() # str with hydrologic data source
    estimate_dam = estimate_dam_height_var.get() # boolean

    # get the numbers of all folders in results folder

    if dropdown.get() == "All Dams":
        dam_strs = successful_runs(results_dir)
        dam_ints = sorted([int(d) for d in dam_strs])
        for dam_id in dam_ints:
            print(f"Analyzing Dam No. {dam_id}")
            dam_i = Dam(int(dam_id), database_csv, selected_model, estimate_dam)
            # dam_i.plot_rating_curves()
            # xs_fig = dam_i.plot_cross_sections()
            # dam_i.plot_all_curves()
            print("Onto the next one! :)")
            # dam_i.plot_map()

            if display_cross_section.get():
                plot_cross_section(dam_i)
            if display_rating_curves.get():
                plot_rating_curves(dam_i)
            if display_map.get():
                plot_map(dam_i)
            if display_wsp.get():
                plot_wsp(dam_i)
    else:
        dam_i = Dam(int(dropdown.get()), database_csv, selected_model, estimate_dam)
        if display_cross_section.get():
            plot_cross_section(dam_i)
        if display_rating_curves.get():
            plot_rating_curves(dam_i)
        if display_map.get():
            plot_map(dam_i)
        if display_wsp.get():
            plot_wsp(dam_i)


# GUI setup
root = tk.Tk()
root.title("ARC Low-Head Dam Analysis")
root.geometry("600x800")

# Database label
section_label = tk.Label(root, text="Database Information", font=("Arial", 12, "bold"))
section_label.pack(pady=(15, 0))

# --- Database Selection ---
database_frame = tk.Frame(root)
database_frame.pack(pady=10, padx=10, fill=tk.X)

tk.Button(database_frame, text="Select Database File", command=select_csv_file).pack(side=tk.LEFT)
database_entry = tk.Entry(database_frame)
database_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

# --- Results Directory Selection ---
results_frame = tk.Frame(root)
results_frame.pack(pady=10, padx=10, fill=tk.X)

tk.Button(results_frame, text="Select Results Folder", command=select_results_dir).pack(side=tk.LEFT)
results_entry = tk.Entry(results_frame)
results_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

# Hydraulics and Hydrology label
section_label = tk.Label(root, text="Hydraulic Data Information", font=("Arial", 12, "bold"))
section_label.pack(pady=(15, 0))

# --- Model Selection Dropdown ---
model_label = tk.Label(root, text="Hydrologic Data Source:")
model_label.pack(pady=(5, 0))

model_var = tk.StringVar()
model_dropdown = ttk.Combobox(root, textvariable=model_var, state="readonly")
model_dropdown['values'] = ("USGS", "GEOGLOWS", "National Water Model")
model_dropdown.current(0)  # default selection
model_dropdown.pack(pady=(0, 10))

# Create a Tkinter Boolean variable to store the checkbox state
estimate_dam_height_var = tk.BooleanVar()
estimate_dam_height_var.set(False)  # default is unchecked
checkbox = ttk.Checkbutton(root, text="Estimate Dam Height?", variable=estimate_dam_height_var)
checkbox.pack()  # or .grid(row=..., column=...) depending on your layout


# Output label
section_label = tk.Label(root, text="Output Information", font=("Arial", 12, "bold"))
section_label.pack(pady=(15, 0))

# Bind entry box changes to update dropdown
# noinspection PyTypeChecker
results_entry.bind("<FocusOut>", update_dropdown)
# noinspection PyTypeChecker
results_entry.bind("<Return>", update_dropdown)

# Dropdown
dropdown_label = tk.Label(root, text="Dam(s) to Analyze:")
dropdown_label.pack()
dropdown = ttk.Combobox(root, state="readonly")
dropdown.pack(pady=5)

# --- Display Graphs Label and Checkboxes ---
section_label = tk.Label(root, text="Select Figures to Display:", font=("Arial", 10))
section_label.pack(pady=(15, 0))

# cross-section checkbox
display_cross_section = tk.BooleanVar()
display_cross_section.set(True)# default checked
display_checkbox = ttk.Checkbutton(root, text="Cross-Sections", variable=display_cross_section)
display_checkbox.pack(pady=(5, 10))

# rating curves checkbox
display_rating_curves = tk.BooleanVar()
display_rating_curves.set(True)# default checked
display_checkbox = ttk.Checkbutton(root, text="Rating Curves", variable=display_rating_curves)
display_checkbox.pack(pady=(5, 10))

# map checkbox
display_map = tk.BooleanVar()
display_map.set(True)# default checked
display_checkbox = ttk.Checkbutton(root, text="Dam Location", variable=display_map)
display_checkbox.pack(pady=(5, 10))

# water surface checkbox
display_wsp = tk.BooleanVar()
display_wsp.set(True)# default checked
display_checkbox = ttk.Checkbutton(root, text="Water Surface Profile", variable=display_wsp)
display_checkbox.pack(pady=(5, 10))


# --- Run function button ---
run_button = tk.Button(root, text="Process ARC Data", command=process_ARC, height=2, width=20)
run_button.pack(pady=20)

# --- Final Graph Button ---
run_button = tk.Button(root, text="Create Summary Figures", command=plot_shj_estimates, height=2, width=20)
run_button.pack(pady=20)

root.mainloop()
