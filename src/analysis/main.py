import os
import ast
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
import geopandas as gpd
from classes import Dam
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading  # Import threading


# noinspection PyBroadException,PyTypeChecker
def update_dropdown():
    results_dir = results_entry.get()
    if os.path.isdir(results_dir):
        try:
            dam_strs = successful_runs(results_dir)
            dams = sorted([int(d) for d in dam_strs])
            dams.insert(0, "All Dams")
            dropdown['values'] = dams
            if dams:
                dropdown.set(dams[0])  # select first item
            status_var.set(f"Found {len(dams) - 1} processed dams.")
        except Exception as e:
            dropdown.set("Error finding dams")
            status_var.set(f"Error reading results folder: {e}")
    else:
        dropdown.set("Invalid results path")


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
        update_dropdown()  # <-- Updated here for immediate feedback


def plot_shj_estimates():
    # --- This is the function that will run in a separate thread ---
    def threaded_plot_shj():
        try:
            status_var.set("Starting to generate summary figures...")
            run_button.config(state=tk.DISABLED)
            summary_button.config(state=tk.DISABLED)

            lhd_df_path = database_entry.get()
            if not os.path.exists(lhd_df_path):
                messagebox.showerror("Error", f"Database file not found:\n{lhd_df_path}")
                return

            lhd_df = pd.read_csv(lhd_df_path)

            # Create new window
            win = tk.Toplevel()
            win.title("All Summary Figures")
            win.geometry("1200x600")

            # Create a canvas with scrollbar
            canvas = tk.Canvas(win)
            scrollbar = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
            scroll_frame = ttk.Frame(canvas)

            scroll_frame.bind("<Configure>", lambda error: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            plot_generated = False
            for i in range(1, 5):
                status_var.set(f"Processing Cross-Section {i}...")
                # Define the columns for the current iteration
                cols_to_check = [f'y_t_{i}', f'y_flip_{i}', f'y_2_{i}', f's_{i}']

                # Drop rows where any of these essential columns are NaN or empty
                filtered_df = lhd_df.dropna(subset=cols_to_check).copy()

                # If after filtering, the dataframe is empty, skip to the next iteration
                if filtered_df.empty:
                    print(f"No data available for cross-section {i}. Skipping plot.")
                    continue

                # Helper function to safely evaluate string lists
                def safe_literal_eval(item):
                    try:
                        # Ensure item is a string and looks like a list before evaluating
                        if pd.notna(item) and isinstance(item, str) and item.strip().startswith('['):
                            return ast.literal_eval(item)
                    except (ValueError, SyntaxError):
                        # Return an empty list if evaluation fails
                        return []
                    return []

                # Process the filtered data
                y_t_strs = filtered_df[f'y_t_{i}'].to_list()
                y_flip_strs = filtered_df[f'y_flip_{i}'].to_list()
                y_2_strs = filtered_df[f'y_2_{i}'].to_list()
                slopes = filtered_df[f's_{i}'].to_list()
                dam_ids = filtered_df['ID'].tolist()

                y_t_list = [num for item in y_t_strs for num in safe_literal_eval(item)]
                y_flip_list = [num for item in y_flip_strs for num in safe_literal_eval(item)]
                y_2_list = [num for item in y_2_strs for num in safe_literal_eval(item)]

                nested_list = [safe_literal_eval(item) for item in y_t_strs]

                # Check if nested_list is empty after processing
                if not any(nested_list):
                    print(f"All rows for cross-section {i} contained empty lists. Skipping plot.")
                    continue

                expanded_slopes = [val for val, group in zip(slopes, nested_list) for _ in range(len(group))]
                expanded_ids = [val for val, group in zip(dam_ids, nested_list) for _ in range(len(group))]
                # --- MODIFICATION END ---

                df = pd.DataFrame({
                    'slope': expanded_slopes,
                    'y_t': y_t_list,
                    'y_flip': y_flip_list,
                    'y_2': y_2_list,
                    'dam_id': expanded_ids
                })

                if df.empty:
                    print(f"DataFrame is empty for cross-section {i} after processing. Skipping plot.")
                    continue

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
                if shade and len(df) > 0:
                    ax.axvspan(start_idx - 0.5, len(df) - 0.5, color='gray', alpha=0.1)

                ax.set_xticks(x_vals)
                ax.set_xticklabels(x_labels, rotation=90)
                ax.set_xlabel('Slope')
                ax.set_ylabel('Depth (ft)')
                ax.grid(True, axis='y', linestyle='--', alpha=0.5)
                ax.set_title(f"Summary of Results from Cross-Section No. {i}")
                # fig.savefig(f'./Summary of Results from Cross-Section No. {i}.png')
                fig.tight_layout()

                fig_canvas = FigureCanvasTkAgg(fig, master=scroll_frame)
                fig_canvas.draw()
                fig_canvas.get_tk_widget().pack(padx=10, pady=10, fill="both", expand=True)
                plot_generated = True

            if not plot_generated:
                status_var.set("No data found to generate summary plots.")
                win.destroy()
            else:
                status_var.set("Summary figures generated successfully.")
        except Exception as e:
            status_var.set(f"Error plotting summary: {e}")
            messagebox.showerror("Plotting Error", f"Failed to generate summary plots:\n{e}")
        finally:
            # Re-enable buttons when done
            run_button.config(state=tk.NORMAL)
            summary_button.config(state=tk.NORMAL)

    # --- Start the thread ---
    threading.Thread(target=threaded_plot_shj, daemon=True).start()


def plot_cross_section(dam):
    # Create new window
    win = tk.Toplevel()
    win.title(f"All Cross-Sections (Dam {dam.id})")
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
    win.title(f"All Rating Curves (Dam {dam.id})")
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
    plot_window.title(f"Dam Map (Dam {dam.id})")
    plot_window.geometry("1200x1000")

    # Embed the figure in the Tkinter window
    canvas = FigureCanvasTkAgg(map_fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def plot_wsp(dam):
    wsp_fig = dam.plot_water_surface()
    plot_window = tk.Toplevel(root)
    plot_window.title(f"Water Surface Profile (Dam {dam.id})")
    plot_window.geometry("1200x1000")
    # Embed the figure in the Tkinter window
    canvas = FigureCanvasTkAgg(wsp_fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def plot_fdc(dam):
    # Create new window
    win = tk.Toplevel()
    win.title(f"All Flow-Duration Curves (Dam {dam.id})")
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
        rc_fig = cross_section.create_combined_fdc()
        fig_canvas = FigureCanvasTkAgg(rc_fig, master=scroll_frame)
        fig_canvas.draw()
        fig_canvas.get_tk_widget().pack(padx=10, pady=10, fill="both", expand=True)


def successful_runs(results_dir):
    dam_nos = [str(d) for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]

    successes = []

    for lhd_id in dam_nos:
        run_results_dir = os.path.join(results_dir, lhd_id)
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
            print(f"Error reading GPKG for {lhd_id}: {e}")
            # Do not return False, just skip this dam
            continue
    return successes


def process_ARC():
    # --- This is the wrapper function that will run in a separate thread ---
    def threaded_process_arc():
        try:
            status_var.set("Starting to process ARC data...")
            run_button.config(state=tk.DISABLED)
            summary_button.config(state=tk.DISABLED)

            # project_dir holds all the project files
            database_csv = database_entry.get()
            results_dir = results_entry.get()
            selected_model = model_var.get()  # str with hydrologic data source
            estimate_dam = estimate_dam_height_var.get()  # boolean

            if not os.path.exists(database_csv):
                messagebox.showerror("Error", f"Database file not found:\n{database_csv}")
                return
            if not os.path.isdir(results_dir):
                messagebox.showerror("Error", f"Results directory not found:\n{results_dir}")
                return

            selected_dam = dropdown.get()

            # --- Logic for plotting only a single dam's figs ---
            plot_figs_on_run = (selected_dam != "All Dams")

            if selected_dam == "All Dams":
                dam_strs = successful_runs(results_dir)
                dam_ints = sorted([int(d) for d in dam_strs])
                total_dams = len(dam_ints)
                for i, dam_id in enumerate(dam_ints):
                    status_var.set(f"Processing Dam {dam_id} ({i + 1} of {total_dams})...")
                    print(f"Analyzing Dam No. {dam_id}")
                    dam_i = Dam(int(dam_id), database_csv, selected_model, estimate_dam)
                    # --- When "All Dams" is selected, just save figs, don't show ---
                    for xs in dam_i.cross_sections:
                        xs.plot_cross_section()  # Saves the fig
                    for xs in dam_i.cross_sections[1:]:
                        xs.create_combined_fig()  # Saves the fig
                        xs.create_combined_fdc()  # Saves the fig
                    dam_i.plot_map()  # Saves the fig
                    dam_i.plot_water_surface()  # Saves the fig
                    print("Onto the next one! :)")

            else:  # --- Logic for a single dam ---
                status_var.set(f"Processing single Dam {selected_dam}...")
                dam_i = Dam(int(selected_dam), database_csv, selected_model, estimate_dam)

                # --- Show plots if checked, as it's just one dam ---
                if plot_figs_on_run:
                    if display_cross_section.get():
                        plot_cross_section(dam_i)
                    if display_rating_curves.get():
                        plot_rating_curves(dam_i)
                    if display_map.get():
                        plot_map(dam_i)
                    if display_wsp.get():
                        plot_wsp(dam_i)
                    if display_fdc.get():
                        plot_fdc(dam_i)
                # Also save the figures, regardless of whether they were shown
                for xs in dam_i.cross_sections:
                    xs.plot_cross_section()  # Saves the fig
                for xs in dam_i.cross_sections[1:]:
                    xs.create_combined_fig()  # Saves the fig
                    xs.create_combined_fdc()  # Saves the fig
                dam_i.plot_map()  # Saves the fig
                dam_i.plot_water_surface()  # Saves the fig

            status_var.set("Processing complete.")

        except Exception as e:
            status_var.set(f"Error during processing: {e}")
            messagebox.showerror("Processing Error", f"An error occurred:\n{e}")
        finally:
            # Re-enable buttons when done
            run_button.config(state=tk.NORMAL)
            summary_button.config(state=tk.NORMAL)

    # --- Start the thread ---
    threading.Thread(target=threaded_process_arc, daemon=True).start()


# GUI setup
root = tk.Tk()
root.title("ARC Low-Head Dam Analysis")
root.geometry("600x800")

# --- Database and Results Paths ---
path_frame = ttk.LabelFrame(root, text="Database and Results Paths")
path_frame.pack(pady=10, padx=10, fill=tk.X)
path_frame.columnconfigure(1, weight=1)

ttk.Button(path_frame, text="Select Database File", command=select_csv_file).grid(row=0, column=0, padx=5, pady=5,
                                                                                  sticky=tk.W)
database_entry = ttk.Entry(path_frame)
database_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

ttk.Button(path_frame, text="Select Results Folder", command=select_results_dir).grid(row=1, column=0, padx=5, pady=5,
                                                                                      sticky=tk.W)
results_entry = ttk.Entry(path_frame)
results_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

# --- Analysis Settings ---
settings_frame = ttk.LabelFrame(root, text="Analysis Settings")
settings_frame.pack(pady=10, padx=10, fill=tk.X)
settings_frame.columnconfigure(1, weight=1)

ttk.Label(settings_frame, text="Hydrologic Data Source:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
model_var = tk.StringVar()
model_dropdown = ttk.Combobox(settings_frame, textvariable=model_var, state="readonly",
                              values=("USGS", "GEOGLOWS", "National Water Model"))
model_dropdown.current(0)  # default selection
model_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

estimate_dam_height_var = tk.BooleanVar(value=False)
checkbox = ttk.Checkbutton(settings_frame, text="Estimate Dam Height?", variable=estimate_dam_height_var)
checkbox.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)

ttk.Label(settings_frame, text="Dam(s) to Analyze:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
dropdown = ttk.Combobox(settings_frame, state="readonly")
dropdown.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)

# --- Figure Display Options ---
figure_frame = ttk.LabelFrame(root, text="Select Figures to Display (for Single Dam Analysis)")
figure_frame.pack(pady=10, padx=10, fill=tk.X)
figure_frame.columnconfigure(0, weight=1)
figure_frame.columnconfigure(1, weight=1)

display_cross_section = tk.BooleanVar(value=True)
display_checkbox = ttk.Checkbutton(figure_frame, text="Cross-Sections", variable=display_cross_section)
display_checkbox.grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)

display_rating_curves = tk.BooleanVar(value=True)
display_checkbox = ttk.Checkbutton(figure_frame, text="Rating Curves", variable=display_rating_curves)
display_checkbox.grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)

display_map = tk.BooleanVar(value=True)
display_checkbox = ttk.Checkbutton(figure_frame, text="Dam Location", variable=display_map)
display_checkbox.grid(row=2, column=0, padx=5, pady=2, sticky=tk.W)

display_wsp = tk.BooleanVar(value=True)
display_checkbox = ttk.Checkbutton(figure_frame, text="Water Surface Profile", variable=display_wsp)
display_checkbox.grid(row=0, column=1, padx=5, pady=2, sticky=tk.W)

display_fdc = tk.BooleanVar(value=True)
display_checkbox = ttk.Checkbutton(figure_frame, text="Flow-Duration Curve", variable=display_fdc)
display_checkbox.grid(row=1, column=1, padx=5, pady=2, sticky=tk.W)

# --- Run Buttons Frame ---
button_frame = ttk.Frame(root)
button_frame.pack(pady=20, fill=tk.X, padx=10)
button_frame.columnconfigure(0, weight=1)
button_frame.columnconfigure(1, weight=1)

run_button = ttk.Button(button_frame, text="Process ARC Data", command=process_ARC)
run_button.grid(row=0, column=0, padx=5, ipady=5, sticky=tk.EW)

summary_button = ttk.Button(button_frame, text="Create Summary Figures", command=plot_shj_estimates)
summary_button.grid(row=0, column=1, padx=5, ipady=5, sticky=tk.EW)

# --- Status Bar ---
status_var = tk.StringVar()
status_var.set("Ready.")
status_label = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
status_label.pack(side=tk.BOTTOM, fill=tk.X, ipady=2)

root.mainloop()