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
"""
    these imports were for plotting all the dams in indiana
"""
# from shapely.geometry import Point
# import contextily as ctx
# from itertools import zip_longest
# from matplotlib_scalebar.scalebar import ScaleBar


def add_north_arrow(ax, x=0.08, y=0.15, arrow_length=0.07, text='N', color='black', outline_color='white'):
    """
    Adds a cleaner north arrow to a matplotlib axes object.

    ax: The axes object.
    x, y: Relative (0-1) position of the arrow base.
    arrow_length: Relative length of the arrow (0-1).
    text: The text to display (e.g., 'N').
    color: Arrow and text color.
    outline_color: Color for the arrow's outline for contrast.
    """
    from matplotlib.patches import FancyArrow

    # Draw outline arrow first (slightly bigger)
    arrow_outline = FancyArrow(x, y, 0, arrow_length,
                               transform=ax.transAxes,
                               width=0.015,
                               length_includes_head=True,
                               head_width=0.03,
                               head_length=arrow_length * 0.25,
                               facecolor=outline_color,
                               edgecolor=outline_color,
                               zorder=5)
    ax.add_patch(arrow_outline)

    # Draw main arrow on top
    arrow_main = FancyArrow(x, y, 0, arrow_length,
                            transform=ax.transAxes,
                            width=0.01,
                            length_includes_head=True,
                            head_width=0.025,
                            head_length=arrow_length * 0.25,
                            facecolor=color,
                            edgecolor=color,
                            zorder=6)
    ax.add_patch(arrow_main)

    # Add text above the arrow tip
    ax.text(x, y + arrow_length + 0.01, text,
            transform=ax.transAxes,
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=16,
            fontweight='bold',
            color=color,
            zorder=7)



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
        fig, ax = plt.subplots(figsize=(13, 5))
        cap_width = 0.2

        legend_map = {'blue': 'Predicted Type A',
                      'red': 'Predicted Type C',
                      'green': 'Predicted Type D'}

        # Track which legend entries we've already added
        added_labels = set()

        for x, y, y2, y_flip in zip(x_vals, tailwater, conjugate, flip):
            if y2 < y < y_flip:
                c = 'red'  # Type C
            elif y <= y2:
                c = 'blue'  # Type A
            else:
                c = 'green'  # Type D

            label = legend_map[c] if c not in added_labels else None
            if label:
                added_labels.add(c)

            ax.vlines(x, y2, y_flip, color='black', linewidth=1)
            ax.hlines(y2, x - cap_width, x + cap_width, color='black', linewidth=1)
            ax.hlines(y_flip, x - cap_width, x + cap_width, color='black', linewidth=1)
            ax.scatter(x, y, color=c, marker='x', zorder=3, label=label)

        current_id = None
        start_idx = 0
        shade = True  # start with shading
        label_y_position = ax.get_ylim()[1] * 0.95  # Position labels near the top

        for idx, dam in enumerate(df['dam_id']):
            if dam != current_id:
                if current_id is not None:
                    # --- Add Dam ID Label ---
                    mid_idx = start_idx + (idx - 1 - start_idx) / 2  # Calculate middle index for the previous group
                    ax.text(mid_idx, label_y_position, f"Dam {current_id}",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
                    # --- End Add Dam ID Label ---

                    # Apply shading to the previous group if needed
                    if shade:
                        ax.axvspan(start_idx - 0.5, idx - 0.5, color='gray', alpha=0.1)

                # Reset for the new dam group
                current_id = dam
                start_idx = idx
                shade = not shade  # Alternate shading

            # Handle the final dam's label and shading
        if len(df) > 0 and current_id is not None:
            # --- Add Final Dam ID Label ---
            mid_idx = start_idx + (len(df) - 1 - start_idx) / 2
            ax.text(mid_idx, label_y_position, f"Dam {current_id}",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
            # --- End Add Final Dam ID Label ---
            if shade:  # Apply shading if needed for the last group
                ax.axvspan(start_idx - 0.5, len(df) - 0.5, color='gray', alpha=0.1)

        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_labels, rotation=90)
        ax.set_xlabel('Slope')
        ax.set_ylabel('Depth (ft)')
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.set_title(f"Summary of Results from Cross-Section No. {i}")
        # Get the directory of the database file
        database_path = database_entry.get()
        database_dir = os.path.dirname(database_path)
        save_path = os.path.join(database_dir, f'Summary of Results from Cross-Section No. {i}.png')

        ax.legend(
            title="Prediction Legend",
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0
        )
        # fig.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on the right

        fig.tight_layout()  # Automatically adjust layout first
        fig.subplots_adjust(bottom=0.2)  # Manually add more space at the bottom

        fig.savefig(save_path, bbox_inches='tight')

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


def plot_fdc(dam):
    # Create new window
    win = tk.Toplevel()
    win.title("All Flow-Duration Curves")
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
            if display_fdc.get():
                plot_fdc(dam_i)
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
        if display_fdc.get():
            plot_fdc(dam_i)


# def plot_all_dams_map():
#     """
#     Plots all dams from the database on a map, colored by predicted jump type.
#     Prioritizes showing Type C (red) if any fatality flow predicts it for a dam.
#     """
#     # --- Ask for Indiana Shapefile ---
#     indiana_shapefile_path = filedialog.askopenfilename(
#         title="Select Indiana Shapefile",
#         filetypes=[("Shapefiles", "*.shp")]
#     )
#     if not indiana_shapefile_path:
#         print("No Indiana shapefile selected. Skipping overlay.")
#         indiana_gdf = None
#     else:
#         try:
#             indiana_gdf = gpd.read_file(indiana_shapefile_path)
#             # Reproject to Web Mercator (same as basemap and dam points)
#             indiana_gdf = indiana_gdf.to_crs(epsg=3857)
#             print(f"Loaded Indiana shapefile: {indiana_shapefile_path}")
#         except Exception as e:
#             print(f"Error loading or reprojecting Indiana shapefile: {e}")
#             indiana_gdf = None  # Proceed without the shapefile if loading fails
#     # --- End Shapefile Handling ---
#
#     try:
#         lhd_df = pd.read_csv(database_entry.get())
#     except Exception as e:
#         print(f"Error reading database CSV: {e}")
#         # Optionally show an error message to the user via messagebox
#         return
#
#     # --- Create lists for each type ---
#     coords_a, coords_c, coords_d, coords_unavail = [], [], [], []
#
#     for _, row in lhd_df.iterrows():
#         lon, lat = row['longitude'], row['latitude']
#         point = Point(lon, lat)
#
#         found_any = False
#         for i in range(1, 5):  # check y_t_1 ... y_t_4
#             y_t_list = ast.literal_eval(row[f'y_t_{i}']) if pd.notna(row.get(f'y_t_{i}')) else []
#             y_2_list = ast.literal_eval(row[f'y_2_{i}']) if pd.notna(row.get(f'y_2_{i}')) else []
#             y_flip_list = ast.literal_eval(row[f'y_flip_{i}']) if pd.notna(row.get(f'y_flip_{i}')) else []
#
#             if not y_t_list:
#                 continue
#
#             # Check types for this cross-section
#             if any(y2 < y < y_flip for y, y2, y_flip in
#                    zip_longest(y_t_list, y_2_list, y_flip_list, fillvalue=float('nan'))):
#                 coords_c.append(point)
#                 found_any = True
#             if any(y >= y_flip for y, y_flip in zip_longest(y_t_list, y_flip_list, fillvalue=float('nan'))):
#                 coords_d.append(point)
#                 found_any = True
#             if any(y <= y2 for y, y2 in zip_longest(y_t_list, y_2_list, fillvalue=float('nan'))):
#                 coords_a.append(point)
#                 found_any = True
#
#         if not found_any:
#             coords_unavail.append(point)
#
#     # --- Create GeoDataFrames for each type ---
#     def make_gdf(coords):
#         if coords:
#             return gpd.GeoDataFrame(geometry=coords, crs="EPSG:4326").to_crs(epsg=3857)
#         else:
#             # Create an empty GeoDataFrame with the same CRS
#             return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326").to_crs(epsg=3857)
#
#     gdf_a = make_gdf(coords_a)
#     gdf_c = make_gdf(coords_c)
#     gdf_d = make_gdf(coords_d)
#     print(len(gdf_d))
#     gdf_unavail = make_gdf(coords_unavail)
#
#     fig, ax = plt.subplots(figsize=(10, 12))
#
#     # --- Plot Indiana boundary first ---
#     if indiana_gdf is not None:
#         indiana_gdf.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5, zorder=1)
#
#     # --- Set extent before basemap ---
#     if indiana_gdf is not None:
#         minx, miny, maxx, maxy = indiana_gdf.total_bounds
#     else:
#         # Fallback if no shapefile
#         all_gdfs = [gdf for gdf in [gdf_a, gdf_c, gdf_d, gdf_unavail] if gdf is not None and not gdf.empty]
#         if all_gdfs:
#             minx, miny, maxx, maxy = gpd.GeoSeries(all_gdfs).total_bounds
#         else:
#             # Default extent if no data at all
#             minx, miny, maxx, maxy = -9777994, 4838600, -9435345, 5122171  # Approx Indiana in Web Mercator
#
#     ax.set_xlim(minx - 20_000, maxx + 20_000)
#     ax.set_ylim(miny - 20_000, maxy + 20_000)
#
#     # --- Basemap below everything ---
#     try:
#         ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zorder=0)
#     except Exception as e:
#         print(f"Could not add basemap: {e}")
#
#     # Plot unavailable first so it doesn’t get fully covered
#     if gdf_unavail is not None and not gdf_unavail.empty:
#         gdf_unavail.plot(ax=ax, color='white', marker='x', markersize=40, alpha=0.6, label='Unavailable', zorder=1)
#
#     # Then plot the rest
#     if gdf_c is not None and not gdf_c.empty:
#         gdf_c.plot(ax=ax, color='red', marker='x', markersize=70, alpha=0.9, label='Type C', zorder=2)
#     if gdf_a is not None and not gdf_a.empty:
#         gdf_a.plot(ax=ax, color='blue', marker='x', markersize=50, alpha=0.8, label='Type A', zorder=3)
#     if gdf_d is not None and not gdf_d.empty:
#         gdf_d.plot(ax=ax, color='lime', marker='x', markersize=60, alpha=0.8, label='Type D', zorder=4)
#
#     # --- Final formatting ---
#
#     # --- Add North Arrow (lower left, matching example style) ---
#     add_north_arrow(ax, x=0.08, y=0.08)  # Adjusted position
#
#     # --- Add Scale Bar (lower center, with white box) ---
#     scalebar = ScaleBar(1, 'm',
#                         location='lower center',
#                         box_alpha=0.85,  # slightly more opaque
#                         box_color='white',
#                         color='black',
#                         frameon=True,  # draw a frame around the box
#                         font_properties={'size': 10, 'weight': 'bold'},
#                         scale_loc='bottom',  # labels below the bar
#                         border_pad=0.5,  # padding inside the box
#                         sep=5,
#                         length_fraction=0.25,
#                         height_fraction=0.02,  # slightly taller for visibility
#                         label_formatter=lambda x, unit: f'{int(x)} {unit}'
#                         )
#
#     if hasattr(scalebar, 'rectangle') and scalebar.rectangle is not None:
#         scalebar.rectangle.set_edgecolor('black')
#         scalebar.rectangle.set_linewidth(1.5)
#
#     ax.add_artist(scalebar)
#
#     ax.set_axis_off()
#     ax.set_title("Map of Low-Head Dams with Predicted Jump Types", fontsize=16, fontweight='bold')
#
#     # --- Adjust legend position to avoid new elements ---
#     ax.legend(title="Prediction Legend", title_fontsize=12, fontsize=10, loc='upper right')
#     database_path = database_entry.get()
#     database_dir = os.path.dirname(database_path)
#     save_path = os.path.join(database_dir, f'Map of Indiana Dams.png')
#     fig.savefig(save_path, bbox_inches='tight')
#     # --- Display in Tkinter window ---
#     map_window = tk.Toplevel(root)
#     map_window.title("All Dams Prediction Map")
#     map_window.geometry("800x1000")  # Adjust size as needed
#
#     canvas = FigureCanvasTkAgg(fig, master=map_window)
#     canvas.draw()
#     canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


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

# flow-duration curve checkbox
display_fdc = tk.BooleanVar()
display_fdc.set(True)# default checked
display_checkbox = ttk.Checkbutton(root, text="Flow-Duration Curve", variable=display_fdc)
display_checkbox.pack(pady=(5, 10))

# --- Run function button ---
run_button = tk.Button(root, text="Process ARC Data", command=process_ARC, height=2, width=20)
run_button.pack(pady=20)

# --- Final Graph Button ---
run_button = tk.Button(root, text="Create Summary Figures", command=plot_shj_estimates, height=2, width=20)
run_button.pack(pady=20)

# # --- All Dams Map Button ---
# map_all_button = tk.Button(root, text="Plot All Dams Map", command=plot_all_dams_map, height=2, width=20)
# map_all_button.pack(pady=10)



root.mainloop()
