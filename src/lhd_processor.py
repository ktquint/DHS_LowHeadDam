import os
import ast
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from hsclient import HydroShare
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# --- Main Application Imports ---
# Import helper modules from your project
try:
    # noinspection PyUnresolvedReferences
    from prep.dam import Dam as PrepDam
    # noinspection PyUnresolvedReferences
    from prep import create_json as cj
    # noinspection PyUnresolvedReferences
    from analysis.classes import Dam as AnalysisDam
except ImportError as e:
    messagebox.showerror("Module Error",
                         f"Could not import a required module: {e}\n\n"
                         "Please make sure you run this script from the 'src' directory, "
                         "and that 'prep' and 'analysis' sub-modules are present.")
    exit()

# Import rathcelon (must be installed in your environment)
try:
    # noinspection PyUnresolvedReferences
    from rathcelon.classes import Dam as RathcelonDam
except ImportError:
    messagebox.showerror("Import Error",
                         "Could not find 'rathcelon' package. \n"
                         "Please ensure it is installed in your Python environment.")
    exit()

"""
===================================================================

           TAB 1: PREPARATION & PROCESSING FUNCTIONS

===================================================================
"""

def select_prep_project_dir():
    """Selects the main project directory and auto-populates all paths for Tab 1."""
    project_path = filedialog.askdirectory()
    if not project_path:
        return  # User canceled

    prep_project_entry.delete(0, tk.END)
    prep_project_entry.insert(0, project_path)

    try:
        # Find the .xlsx database
        csv_files = [f for f in os.listdir(project_path) if f.endswith('.csv')]
        if not csv_files:
            messagebox.showwarning("No Database", f"No .csv database file found in:\n{project_path}")
            return

        database_path = os.path.join(project_path, csv_files[0])
        prep_database_entry.delete(0, tk.END)
        prep_database_entry.insert(0, database_path)

        # Auto-populate other paths
        results_path = os.path.join(project_path, "LHD_Results")

        prep_dem_entry.delete(0, tk.END)
        prep_dem_entry.insert(0, results_path)  # download_dem expects the base results folder

        prep_strm_entry.delete(0, tk.END)
        prep_strm_entry.insert(0, os.path.join(project_path, "LHD_STRMs"))

        prep_results_entry.delete(0, tk.END)
        prep_results_entry.insert(0, results_path)

        # Auto-populate both the creation path and the run path for the JSON
        json_path = os.path.splitext(database_path)[0] + '.json'
        prep_json_entry.delete(0, tk.END)
        prep_json_entry.insert(0, json_path)
        rathcelon_json_entry.delete(0, tk.END)
        rathcelon_json_entry.insert(0, json_path)

        status_var.set("Project paths loaded.")

    except IndexError:
        messagebox.showerror("Error", f"No .csv database file found in:\n{project_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load project paths: {e}")


def select_prep_database_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        prep_database_entry.delete(0, tk.END)
        prep_database_entry.insert(0, file_path)

        prep_json_entry.delete(0, tk.END)
        prep_json_entry.insert(0, os.path.splitext(file_path)[0] + '.json')


def select_prep_dem_dir():
    dem_path = filedialog.askdirectory()
    if dem_path:
        prep_dem_entry.delete(0, tk.END)
        prep_dem_entry.insert(0, dem_path)


def select_prep_strm_dir():
    strm_path = filedialog.askdirectory()
    if strm_path:
        prep_strm_entry.delete(0, tk.END)
        prep_strm_entry.insert(0, strm_path)


def select_prep_results_dir():
    results_path = filedialog.askdirectory()
    if results_path:
        prep_results_entry.delete(0, tk.END)
        prep_results_entry.insert(0, results_path)


def select_prep_json_file():
    json_path = filedialog.asksaveasfilename(filetypes=[("JSON files", "*.json")], defaultextension=".json")
    if json_path:
        prep_json_entry.delete(0, tk.END)
        prep_json_entry.insert(0, json_path)


def select_rathcelon_json_file():
    """NEW function to select the JSON file for the 'Run' step."""
    json_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
    if json_path:
        rathcelon_json_entry.delete(0, tk.END)
        rathcelon_json_entry.insert(0, json_path)


def threaded_prepare_data():
    """
    This function now prepares data and creates the input files.
    """
    try:
        # --- 1. Get all values from GUI ---
        lhd_csv = prep_database_entry.get()
        hydrography = prep_hydro_var.get()
        dem_resolution = prep_dd_var.get()
        hydrology = prep_logy_var.get()
        dem_folder = prep_dem_entry.get()
        strm_folder = prep_strm_entry.get()
        results_folder = prep_results_entry.get()

        # --- 2. Validate inputs ---
        if not os.path.exists(lhd_csv):
            messagebox.showerror("Error", f"Database file not found:\n{lhd_csv}")
            return
        if not all(
                [lhd_csv, hydrography, dem_resolution, hydrology, dem_folder, strm_folder, results_folder]):
            messagebox.showwarning("Missing Info", "Please fill out all path and setting fields.")
            return

        status_var.set("Inputs validated. Starting data prep...")

        # --- 3. Create directories ---
        os.makedirs(dem_folder, exist_ok=True)
        os.makedirs(strm_folder, exist_ok=True)
        os.makedirs(results_folder, exist_ok=True)

        lhd_df = pd.read_csv(lhd_csv)
        final_df = lhd_df.copy()

        # We just use the first row as an example to get the attribute names
        try:
            sample_dam_dict = PrepDam(**lhd_df.iloc[0].to_dict()).__dict__
            cols_to_update = [key for key in sample_dam_dict.keys() if key in final_df.columns]

            # Change the type of these columns to 'object' in the DataFrame
            for col in cols_to_update:
                if final_df[col].dtype != 'object':
                    final_df[col] = final_df[col].astype(object)

            status_var.set("DataFrame types prepared.")
        except Exception as e:
            print(f"Warning: Could not pre-set DataFrame dtypes: {e}")

        hydroshare_id = "88759266f9c74df8b5bb5f52d142ba8e"

        # --- 4. Load NWM data once --- #
        nwm_ds = None
        # Get the absolute path to the directory containing lhd_processor.py
        script_dir = os.path.dirname(os.path.realpath(__file__))
        # Join that directory path with the relative path to your file
        nwm_parquet = os.path.join(script_dir, 'data', 'nwm_daily_retrospective.parquet')

        if hydrology == 'National Water Model':

            # --- check if nwm parquet file exists --- #
            if not os.path.exists(nwm_parquet):
                try:
                    messagebox.showinfo("Downloading Data",
                                        "The NWM Parquet file is missing.\n"
                                        "The program will now download it from HydroShare.\n"
                                        "This may take a moment.")
                    # let's make the dir then
                    data_path = os.path.join(script_dir, 'data')
                    os.makedirs(data_path, exist_ok=True)

                    # download the parquet
                    hs = HydroShare()
                    resource = hs.resource(hydroshare_id)
                    resource.file_download(path='nwm_daily_retrospective.parquet',
                                           save_path=nwm_parquet)
                    status_var.set("NWM Parquet download complete.")

                except Exception as e:
                    messagebox.showerror("Download Failed",
                                         f"Failed to automatically download the NWM Parquet file.\n\n"
                                         f"Error: {e}\n\n"
                                         "Please download the file manually and place it in the 'src/data' folder.")
                    status_var.set("ERROR: NWM parquet file download failed.")

            status_var.set("Reading NWM Parquet...")
            # --- Try to load the dataset (if download didn't fail) ---
            if nwm_ds is None:  # Will be None if file existed OR download succeeded
                try:
                    status_var.set("Loading NWM dataset...")
                    nwm_df = pd.read_parquet(nwm_parquet)
                    nwm_ds = nwm_df.to_xarray()
                    status_var.set("NWM dataset loaded.")
                except FileNotFoundError:
                    print(f"ERROR: NWM Parquet file not found at {nwm_parquet}")
                    status_var.set("ERROR: NWM parquet file not found.")
                    nwm_ds = None
                except Exception as e:
                    print(f"Could not open NWM dataset. Error: {e}")
                    status_var.set("Error: Could not load NWM dataset.")
                    nwm_ds = None

        # --- 5. Main Processing Loop ---
        total_dams = len(lhd_df)
        processed_dams_count = 0

        for i, row in lhd_df.iterrows():
            i = int(str(i))
            dam_id = row.get("ID", f"Row_{i + 2}")

            try:
                status_var.set(f"Prep: Dam {dam_id} ({i + 1} of {total_dams})...")

                dam = PrepDam(**row.to_dict())
                dam.assign_hydrology(hydrology)
                dam.assign_hydrography(hydrography)

                status_var.set(f"Dam {dam_id}: Assigning flowlines...")

                # --- Find the VPU map file (and download if missing) ---
                # Define the path to the data folder and the VPU map file
                vpu_data_dir = "./data"  # <--- NEW: Define the save *directory*
                tdx_vpu_map_path = os.path.join(vpu_data_dir, "vpu-boundaries.gpkg")  # <--- Use os.path.join

                vpu_filename = "vpu-boundaries.gpkg"

                if hydrography == 'GEOGLOWS':
                    if not os.path.exists(tdx_vpu_map_path):
                        # File is missing, let's download it from HydroShare
                        try:
                            # Inform the user this is a one-time download
                            messagebox.showinfo("Downloading Data",
                                                "The GEOGLOWS VPU map file is missing.\n"
                                                "The program will now download it from HydroShare.\n")
                            status_var.set("Downloading vpu-boundaries.gpkg...")

                            # Ensure the 'data' directory exists before saving to it
                            os.makedirs(vpu_data_dir, exist_ok=True)

                            # Download the file
                            hs = HydroShare()
                            resource = hs.resource(hydroshare_id)

                            # Pass the *directory* to save_path, not the full file path
                            resource.file_download(path=vpu_filename,
                                                   save_path=vpu_data_dir)

                            status_var.set("VPU map download complete.")

                        except Exception as e:
                            messagebox.showerror("Download Failed",
                                                 f"Failed to automatically download the VPU map file \
                                                 from: HydroShare\n\n"
                                                 f"Error: {e}\n\n"
                                                 "Please download the file manually and place it in the 'data' folder "
                                                 "of your project, then try again.")
                            continue  # Skips this dam

                # 'strm_folder' is the LHD_STRMs path (where NHD/VPU downloads go)
                # 'tdx_vpu_map_path' is the path to our map file (it either existed or was just downloaded)
                dam.assign_flowlines(strm_folder, tdx_vpu_map_path)

                status_var.set(f"Dam {dam_id}: Assigning DEM...")
                dam.assign_dem(dem_folder, dem_resolution)

                if dam.dem_1m is None and dam.dem_3m is None and dam.dem_10m is None:
                    print(f"Skipping Dam No. {dam_id}: DEM assignment failed.")
                    status_var.set(f"Dam {dam_id}: DEM failed. Skipping.")
                    continue

                dam.assign_output(results_folder)

                needs_reach = False
                if hydrology == 'National Water Model':
                    if pd.isna(row.get('dem_baseflow_NWM')) or pd.isna(row.get('fatality_flows_NWM')):
                        needs_reach = True
                elif hydrology == 'GEOGLOWS':
                    if pd.isna(row.get('dem_baseflow_GEOGLOWS')) or pd.isna(row.get('fatality_flows_GEOGLOWS')):
                        needs_reach = True

                if needs_reach:
                    if hydrology == 'National Water Model' and nwm_ds is None:
                        print(f"Skipping flow estimation for Dam No. {dam_id}: NWM dataset not loaded.")
                    else:
                        status_var.set(f"Dam {dam_id}: Creating stream reach...")
                        dam.create_reach(nwm_ds)

                        status_var.set(f"Dam {dam_id}: Estimating baseflow...")
                        dam.est_dem_baseflow()

                        status_var.set(f"Dam {dam_id}: Estimating fatal flows...")
                        dam.est_fatal_flows()

                dam.fdc_to_csv()

                for key, value in dam.__dict__.items():
                    if key in final_df.columns:
                        # Check if the value is a list or numpy array
                        if isinstance(value, (list, np.ndarray)):
                            # Convert it to a string representation before saving
                            final_df.loc[i, key] = str(value)
                        else:
                            # Otherwise, assign it directly
                            final_df.loc[i, key] = value

                processed_dams_count += 1
                print(f'Finished Prep for Dam No. {dam_id}')

            except Exception as e:
                print(f"---" * 20)
                print(f"CRITICAL ERROR preparing Dam No. {dam_id}: {e}")
                print(f"Skipping this dam and moving to the next one.")
                print(f"---" * 20)
                status_var.set(f"Error on Dam {dam_id}: {e}. Skipping.")
                continue

        # --- 6. Final Output Generation (CSV/Excel/JSON) ---
        if processed_dams_count > 0:
            status_var.set("Saving updated database file...")
            final_df.to_csv(lhd_csv, index=False)

            status_var.set("Creating rathcelon_input.json...")
            json_loc = prep_json_entry.get()

            # This function now also returns the dam dictionaries
            cj.rathcelon_input(lhd_csv, json_loc, hydrography, hydrology, nwm_parquet)

            status_var.set(f"Data preparation complete. {processed_dams_count} dams prepped.")
            messagebox.showinfo("Success", f"Data preparation complete.\n{processed_dams_count} dams processed.\n"
                                           f"Input file created at:\n{json_loc}")
        else:
            status_var.set("No dams were pre-processed successfully.")
            messagebox.showwarning("Process Finished", "No new dam data was pre-processed successfully.")

    except Exception as e:
        status_var.set(f"Fatal error: {e}")
        messagebox.showerror("Fatal Error", f"The process failed:\n{e}")
    finally:
        # --- 7. Re-enable button ---
        prep_run_button.config(state=tk.NORMAL)


def threaded_run_rathcelon():
    """
    This new function ONLY runs the Rathcelon analysis using a selected JSON file.
    """
    try:
        # --- 1. Get JSON file path ---
        json_loc = rathcelon_json_entry.get()
        if not os.path.exists(json_loc):
            messagebox.showerror("Error", f"RathCelon input file not found:\n{json_loc}")
            return

        status_var.set(f"Loading input file: {os.path.basename(json_loc)}...")

        # --- 2. Load and parse JSON ---
        with open(json_loc, 'r') as f:
            data = json.load(f)

        dam_dictionaries = data.get("dams", [])
        if not dam_dictionaries:
            status_var.set("No dams found in the selected JSON file.")
            messagebox.showwarning("Empty File", "No dams to process in the selected JSON file.")
            return

        # --- 3. RUN RATHCELON PROCESS ---
        status_var.set(f"Starting RathCelon analysis for {len(dam_dictionaries)} dams...")
        rathcelon_success_count = 0
        total_dams = len(dam_dictionaries)

        for i, dam_dict in enumerate(dam_dictionaries):
            dam_name = dam_dict.get('name', f"Dam {i + 1}")
            try:
                status_var.set(f"RathCelon: Processing Dam {dam_name} ({i + 1} of {total_dams})...")

                dam_i = RathcelonDam(**dam_dict)
                dam_i.process_dam()

                rathcelon_success_count += 1
            except Exception as e:
                print(f"RathCelon failed for dam {dam_name}: {e}")
                status_var.set(f"RathCelon Error on Dam {dam_name}. Skipping.")
                # Continue to the next dam

        status_var.set(f"RathCelon process complete. {rathcelon_success_count} dams processed.")
        messagebox.showinfo("Success", f"RathCelon process complete.\n{rathcelon_success_count} dams processed.")

    except Exception as e:
        status_var.set(f"Fatal error during Rathcelon run: {e}")
        messagebox.showerror("Fatal Error", f"The RathCelon process failed:\n{e}")
    finally:
        # --- 4. Re-enable button ---
        rathcelon_run_button.config(state=tk.NORMAL)


def start_prep_thread():
    """Triggers the preparation thread."""
    prep_run_button.config(state=tk.DISABLED)
    status_var.set("Starting data preparation...")
    threading.Thread(target=threaded_prepare_data, daemon=True).start()


def start_rathcelon_run_thread():
    """Triggers the Rathcelon analysis thread."""
    rathcelon_run_button.config(state=tk.DISABLED)
    status_var.set("Starting RathCelon analysis...")
    threading.Thread(target=threaded_run_rathcelon, daemon=True).start()


"""
===================================================================

            TAB 2: ANALYSIS & VISUALIZATION FUNCTIONS

===================================================================
"""

# --- Global variables for figure carousel ---
current_figure_list = []
current_figure_index = 0
current_figure_canvas = None


def analysis_successful_runs(results_dir, database_csv):
    """
    Finds all dams in the results_dir that have valid output,
    but ONLY checks for dams listed in the database_csv.
    """
    # --- 1. Get dams from CSV database ---
    try:
        lhd_df = pd.read_csv(database_csv)
        if 'ID' not in lhd_df.columns:
            print("Analysis tab: Database CSV must have an 'ID' column.")
            return []
        # Get a clean set of 'ID's as strings
        dam_nos_from_csv = set(pd.to_numeric(lhd_df['ID'], errors='coerce').dropna().astype(int).astype(str).tolist())
    except Exception as e:
        print(f"Error reading database CSV {database_csv}: {e}")
        return []

    if not dam_nos_from_csv:
        print("No dams found in the database CSV.")
        return []

    # --- 2. Check which of those dams have successful runs ---
    successes = []
    for lhd_id in dam_nos_from_csv:
        run_results_dir = os.path.join(results_dir, lhd_id)

        # Skip if the corresponding folder doesn't even exist
        if not os.path.isdir(run_results_dir):
            continue

        # Check for the specific success file
        local_vdt_gpkg = os.path.join(str(run_results_dir), "VDT", f"{lhd_id}_Local_VDT_Database.gpkg")
        if not os.path.exists(local_vdt_gpkg):
            continue

        # Finally, check if the file is valid and not empty
        try:
            local_vdt_gdf = gpd.read_file(local_vdt_gpkg)
            if not local_vdt_gdf.empty:
                successes.append(lhd_id)  # Add the string ID
        except Exception as e:
            print(f"Error reading GPKG for {lhd_id}: {e}")
            continue

    return successes


def update_analysis_dropdown():
    """Updates the dam selection dropdown on the Analysis tab."""
    results_dir = analysis_results_entry.get()
    database_csv = analysis_database_entry.get()

    # --- 1. Validate paths ---
    if not os.path.isdir(results_dir):
        analysis_dam_dropdown['values'] = []
        analysis_dam_dropdown.set("Invalid results path")
        status_var.set("Invalid results path. Please select a valid folder.")
        return

    if not os.path.isfile(database_csv):
        analysis_dam_dropdown['values'] = []
        analysis_dam_dropdown.set("Invalid database path")
        status_var.set("Invalid database .csv. Please select a valid file.")
        return

    try:
        # --- 2. Get successfully run dams that are in the CSV ---
        # This call is now much more efficient
        dam_strs = analysis_successful_runs(results_dir, database_csv)

        # --- 3. Populate dropdown ---
        dams_int = []
        for d in dam_strs:
            try:
                dams_int.append(int(d))
            except ValueError:
                print(f"Analysis tab: Skipping non-numeric success ID '{d}'")

        dams_sorted = sorted(dams_int)
        dams = ["All Dams"] + [str(d) for d in dams_sorted]

        analysis_dam_dropdown['values'] = dams
        if dams:
            analysis_dam_dropdown.set(dams[0])

        status_var.set(f"Found {len(dams) - 1} processed dams from database for analysis.")

    except Exception as e:
        analysis_dam_dropdown['values'] = []
        analysis_dam_dropdown.set("Error")
        status_var.set(f"Error updating dropdown: {e}")


def select_analysis_csv_file():
    file_path = filedialog.askopenfilename(
        title="Select a CSV file",
        filetypes=[("CSV files", "*.csv")]
    )
    if file_path:
        analysis_database_entry.delete(0, tk.END)
        analysis_database_entry.insert(0, file_path)


def select_analysis_results_dir():
    results_path = filedialog.askdirectory()
    if results_path:
        analysis_results_entry.delete(0, tk.END)
        analysis_results_entry.insert(0, results_path)
        update_analysis_dropdown()  # Update dropdown on folder select


# --- Figure Carousel Functions ---

# noinspection PyUnresolvedReferences
def clear_figure_carousel():
    """Hides the figure viewer and clears its contents."""
    global current_figure_list, current_figure_index, current_figure_canvas
    current_figure_list = []
    current_figure_index = 0

    # Destroy the canvas widget if it exists
    if current_figure_canvas:
        current_figure_canvas.get_tk_widget().destroy()
        current_figure_canvas = None

    # Hide the main viewer frame
    analysis_figure_viewer_frame.pack_forget()


def display_figure(index):
    """Displays the figure at the given index in the carousel."""
    global current_figure_list, current_figure_index, current_figure_canvas

    if not current_figure_list:
        clear_figure_carousel()
        return

    # Ensure index is valid
    index = max(0, min(index, len(current_figure_list) - 1))
    current_figure_index = index

    # Clear old canvas
    if current_figure_canvas:
        current_figure_canvas.get_tk_widget().destroy()

    # Get new figure and title
    fig, title = current_figure_list[index]

    # Create new canvas
    current_figure_canvas = FigureCanvasTkAgg(fig, master=analysis_figure_canvas_frame)
    current_figure_canvas.draw()
    current_figure_canvas.get_tk_widget().pack(fill="both", expand=True)

    # Update label and buttons
    analysis_figure_label_var.set(f"{title} ({index + 1} of {len(current_figure_list)})")
    prev_button.config(state=tk.NORMAL if index > 0 else tk.DISABLED)
    next_button.config(state=tk.NORMAL if index < len(current_figure_list) - 1 else tk.DISABLED)


def on_prev_figure():
    display_figure(current_figure_index - 1)


def on_next_figure():
    display_figure(current_figure_index + 1)


def setup_figure_carousel(figures_list):
    """Called by the thread to populate and show the figure carousel."""
    global current_figure_list, current_figure_index

    clear_figure_carousel()  # Clear any previous figures

    if figures_list:
        current_figure_list = figures_list
        current_figure_index = 0
        analysis_figure_viewer_frame.pack(fill="both", expand=True, padx=10, pady=10)  # Show the viewer
        display_figure(0)  # Show the first figure
    else:
        status_var.set("Analysis complete. No figures were selected for display.")


# --- Core Logic Functions (for Tab 2) ---

def threaded_process_ARC():
    """
    Runs the analysis for selected dams in a thread.
    This function ONLY saves data and figures, it does not display them.
    """
    try:
        status_var.set("Starting data processing...")
        analysis_run_button.config(state=tk.DISABLED)
        analysis_summary_button.config(state=tk.DISABLED)

        database_csv = analysis_database_entry.get()
        results_dir = analysis_results_entry.get()
        selected_model = analysis_model_var.get()
        estimate_dam = analysis_estimate_dam_var.get()

        if not os.path.exists(database_csv):
            messagebox.showerror("Error", f"Database file not found:\n{database_csv}")
            return
        if not os.path.isdir(results_dir):
            messagebox.showerror("Error", f"Results directory not found:\n{results_dir}")
            return

        selected_dam = analysis_dam_dropdown.get()

        if selected_dam == "All Dams":
            dam_strs = analysis_successful_runs(results_dir, database_csv)

            dams_int = []
            for d in dam_strs:
                try:
                    dams_int.append(int(d))
                except ValueError:
                    pass  # Skip non-numeric folders

            dam_ints = sorted(dams_int)
            total_dams = len(dam_ints)
            for i, dam_id in enumerate(dam_ints):
                try:
                    status_var.set(f"Analyzing Dam {dam_id} ({i + 1} of {total_dams})...")
                    print(f"Analyzing Dam No. {dam_id}")
                    dam_i = AnalysisDam(int(dam_id), database_csv, selected_model, estimate_dam, results_dir)

                    # When "All Dams" is selected, just save figs
                    for xs in dam_i.cross_sections:
                        plt.close(xs.plot_cross_section())  # Plot and close to save
                    for xs in dam_i.cross_sections[1:]:
                        plt.close(xs.create_combined_fig())  # Plot and close to save
                        plt.close(xs.create_combined_fdc())  # Plot and close to save
                    plt.close(dam_i.plot_map())  # Plot and close to save
                    plt.close(dam_i.plot_water_surface())  # Plot and close to save
                    print("Onto the next one! :)")

                except ValueError as e:
                    if "Invalid flow conditions" in str(e):
                        print(f"---" * 20)
                        print(f"SKIPPING Dam {dam_id}: {e}")
                        print(f"---" * 20)
                        status_var.set(f"Skipping Dam {dam_id} (Invalid flow)")
                        continue  # Move to the next dam
                    else:
                        print(f"---" * 20)
                        print(f"CRITICAL ValueError on Dam {dam_id}: {e}")
                        print(f"---" * 20)
                        status_var.set(f"Error on Dam {dam_id}. Skipping.")
                        continue
                except Exception as e:
                    print(f"---" * 20)
                    print(f"CRITICAL FAILED processing Dam {dam_id}: {e}")
                    print(f"Skipping this dam and moving to the next one.")
                    print(f"---" * 20)
                    status_var.set(f"Error on Dam {dam_id}. Skipping.")
                    continue

        else:  # --- Logic for a single dam ---
            try:  # <--- START OF ERROR HANDLING
                status_var.set(f"Processing single Dam {selected_dam}...")
                dam_i = AnalysisDam(int(selected_dam), database_csv, selected_model, estimate_dam, results_dir)

                # Just save the figures, do not display
                for xs in dam_i.cross_sections:
                    plt.close(xs.plot_cross_section())
                for xs in dam_i.cross_sections[1:]:
                    plt.close(xs.create_combined_fig())
                    plt.close(xs.create_combined_fdc())
                plt.close(dam_i.plot_map())
                plt.close(dam_i.plot_water_surface())
                print(f"Finished processing and saving figures for Dam {selected_dam}.")

            except ValueError as e:
                if "Invalid flow conditions" in str(e):
                    print(f"---" * 20)
                    print(f"ERROR on Dam {selected_dam}: {e}")
                    print(f"---" * 20)
                    status_var.set(f"Failed Dam {selected_dam} (Invalid flow)")
                    messagebox.showerror("Analysis Error", f"Failed to process Dam {selected_dam}:\n{e}")
                else:
                    print(f"---" * 20)
                    print(f"CRITICAL ValueError on Dam {selected_dam}: {e}")
                    print(f"---" * 20)
                    status_var.set(f"Error on Dam {selected_dam}.")
                    messagebox.showerror("Analysis Error", f"A value error occurred on Dam {selected_dam}:\n{e}")

            except Exception as e:  # <--- ADDED ERROR HANDLING
                print(f"---" * 20)
                print(f"CRITICAL FAILED processing Dam {selected_dam}: {e}")
                print(f"---" * 20)
                status_var.set(f"Error on Dam {selected_dam}.")
                messagebox.showerror("Analysis Error", f"A critical error occurred on Dam {selected_dam}:\n{e}")
                # <--- END OF ADDED ERROR HANDLING

        status_var.set("Analysis processing complete.")
        messagebox.showinfo("Success", f"Finished processing data for {selected_dam}.")

    except Exception as e:
        status_var.set(f"Error during analysis: {e}")
        messagebox.showerror("Processing Error", f"An error occurred:\n{e}")
    finally:
        analysis_run_button.config(state=tk.NORMAL)
        analysis_summary_button.config(state=tk.NORMAL)


def threaded_display_dam_figures():
    """
    NEW Function.
    Generates and displays figures for a SINGLE selected dam in the carousel.
    """
    figures_to_display = []
    try:
        status_var.set("Generating figures for display...")
        analysis_display_button.config(state=tk.DISABLED)

        database_csv = analysis_database_entry.get()
        results_dir = analysis_results_entry.get()
        selected_model = analysis_model_var.get()
        estimate_dam = analysis_estimate_dam_var.get()
        selected_dam = analysis_dam_dropdown.get()

        if not os.path.exists(database_csv) or not os.path.isdir(results_dir):
            messagebox.showerror("Error", "Database or Results path is invalid.")
            return

        if selected_dam == "All Dams":
            messagebox.showinfo("Select Dam", "Please select a single dam from the dropdown to display figures.")
            return

        # --- Logic for a single dam ---
        status_var.set(f"Loading Dam {selected_dam} for display...")
        dam_i = AnalysisDam(int(selected_dam), database_csv, selected_model, estimate_dam, results_dir)

        if analysis_display_cross_section.get():
            for xs in dam_i.cross_sections:
                fig = xs.plot_cross_section()
                title = f"Cross Section {xs.index} (Dam {dam_i.id})"
                figures_to_display.append((fig, title))

        if analysis_display_rating_curves.get():
            for xs in dam_i.cross_sections[1:]:
                fig = xs.create_combined_fig()
                title = f"Rating Curve {xs.index} (Dam {dam_i.id})"
                figures_to_display.append((fig, title))

        if analysis_display_map.get():
            fig = dam_i.plot_map()
            title = f"Dam Location (Dam {dam_i.id})"
            figures_to_display.append((fig, title))

        if analysis_display_wsp.get():
            fig = dam_i.plot_water_surface()
            title = f"Water Surface Profile (Dam {dam_i.id})"
            figures_to_display.append((fig, title))

        if analysis_display_fdc.get():
            for xs in dam_i.cross_sections[1:]:
                fig = xs.create_combined_fdc()
                title = f"Flow Duration Curve {xs.index} (Dam {dam_i.id})"
                figures_to_display.append((fig, title))

        status_var.set(f"Generated {len(figures_to_display)} figures for Dam {selected_dam}.")

        # --- After thread is done, call the setup function on the main thread ---
        root.after(0, setup_figure_carousel, figures_to_display)

    except Exception as e:
        status_var.set(f"Error generating figures: {e}")
        messagebox.showerror("Figure Generation Error", f"An error occurred while generating figures:\n{e}")
        # noinspection PyTypeChecker
        root.after(0, clear_figure_carousel) # Clear carousel on error
    finally:
        analysis_display_button.config(state=tk.NORMAL)


def threaded_plot_shj():
    """Runs the summary plot generation in a thread."""
    try:
        status_var.set("Starting to generate summary bar chart...")
        analysis_run_button.config(state=tk.DISABLED)
        analysis_summary_button.config(state=tk.DISABLED)

        lhd_df_path = analysis_database_entry.get()
        if not os.path.exists(lhd_df_path):
            messagebox.showerror("Error", f"Database file not found:\n{lhd_df_path}")
            return

        lhd_df = pd.read_csv(lhd_df_path)

        # --- This creates a new pop-up window ---
        win = tk.Toplevel(root)
        win.title("All Summary Figures")
        win.geometry("1200x600")
        canvas = tk.Canvas(win)
        scrollbar = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        plot_generated = False
        for i in range(1, 5):
            status_var.set(f"Processing Summary for Cross-Section {i}...")
            cols_to_check = [f'y_t_{i}', f'y_flip_{i}', f'y_2_{i}', f's_{i}']
            filtered_df = lhd_df.dropna(subset=cols_to_check).copy()

            if filtered_df.empty:
                print(f"No data available for cross-section {i}. Skipping plot.")
                continue

            def safe_literal_eval(item):
                try:
                    if pd.notna(item) and isinstance(item, str) and item.strip().startswith('['):
                        return ast.literal_eval(item)
                except (ValueError, SyntaxError):
                    return []
                return []

            y_t_strs = filtered_df[f'y_t_{i}'].to_list()
            y_flip_strs = filtered_df[f'y_flip_{i}'].to_list()
            y_2_strs = filtered_df[f'y_2_{i}'].to_list()
            slopes = filtered_df[f's_{i}'].to_list()
            dam_ids = filtered_df['ID'].tolist()

            y_t_list = [num for item in y_t_strs for num in safe_literal_eval(item)]
            y_flip_list = [num for item in y_flip_strs for num in safe_literal_eval(item)]
            y_2_list = [num for item in y_2_strs for num in safe_literal_eval(item)]
            nested_list = [safe_literal_eval(item) for item in y_t_strs]

            if not any(nested_list):
                print(f"All rows for cross-section {i} contained empty lists. Skipping plot.")
                continue

            expanded_slopes = [val for val, group in zip(slopes, nested_list) for _ in range(len(group))]
            expanded_ids = [val for val, group in zip(dam_ids, nested_list) for _ in range(len(group))]

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

            fig = Figure(figsize=(11, 5))  # Use Figure instead of plt.subplots
            ax = fig.add_subplot(111)
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
            shade = True
            for idx, dam in enumerate(df['dam_id']):
                if dam != current_id:
                    if current_id is not None and shade:
                        ax.axvspan(start_idx - 0.5, idx - 0.5, color='gray', alpha=0.1)
                    current_id = dam
                    start_idx = idx
                    shade = not shade
            if shade and len(df) > 0:
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
            plot_generated = True

        if not plot_generated:
            status_var.set("No data found to generate summary plots.")
            win.destroy()
        else:
            status_var.set("Summary bar chart generated successfully.")
    except Exception as e:
        status_var.set(f"Error plotting summary: {e}")
        messagebox.showerror("Plotting Error", f"Failed to generate summary plots:\n{e}")
    finally:
        analysis_run_button.config(state=tk.NORMAL)
        analysis_summary_button.config(state=tk.NORMAL)


def start_analysis_processing():
    """Triggers the analysis processing thread."""
    clear_figure_carousel()  # Clear any old figures
    analysis_run_button.config(state=tk.DISABLED)
    analysis_summary_button.config(state=tk.DISABLED)
    status_var.set("Starting analysis processing...")
    threading.Thread(target=threaded_process_ARC, daemon=True).start()


def start_display_dam_figures_thread():
    """Triggers the NEW display figures thread."""
    clear_figure_carousel()
    analysis_display_button.config(state=tk.DISABLED)
    status_var.set("Starting to generate dam figures...")
    threading.Thread(target=threaded_display_dam_figures, daemon=True).start()


def start_summary_plotting():
    """Triggers the summary plot thread."""
    clear_figure_carousel()  # Clear any old figures
    analysis_run_button.config(state=tk.DISABLED)
    analysis_summary_button.config(state=tk.DISABLED)
    status_var.set("Starting summary bar chart generation...")
    threading.Thread(target=threaded_plot_shj, daemon=True).start()

"""
===================================================================

                    MAIN APPLICATION GUI SETUP
                
===================================================================
"""

root = tk.Tk()
root.title("LHD Control Center")
root.geometry("700x1000")

# --- Style ---
style = ttk.Style()
style.configure("Accent.TButton", font=("Arial", 10, "bold"))
style.configure("TNotebook.Tab", font=("Arial", 10, "bold"))

# --- Notebook (Tabs) ---
notebook = ttk.Notebook(root)
prep_tab = ttk.Frame(notebook)
analysis_tab = ttk.Frame(notebook)

notebook.add(prep_tab, text="  Preparation & Processing  ")
notebook.add(analysis_tab, text="  Analysis & Visualization  ")
notebook.pack(expand=True, fill="both", padx=10, pady=10)

"""
===================================================================
                --- GUI: PREPARATION TAB ---
===================================================================
"""

# --- Frame for Step 1: Data Preparation ---
prep_data_frame = ttk.LabelFrame(prep_tab, text="Step 1: Prepare Data and Create Input File")
prep_data_frame.pack(pady=10, padx=10, fill=tk.X)

# --- Project Directory Selection ---
prep_project_frame = ttk.Frame(prep_data_frame)
prep_project_frame.pack(pady=5, padx=10, fill=tk.X)
prep_project_frame.columnconfigure(1, weight=1)
ttk.Button(prep_project_frame, text="Select Project Folder", command=select_prep_project_dir).grid(row=0, column=0,
                                                                                                   padx=5, pady=5,
                                                                                                   sticky=tk.W)
prep_project_entry = ttk.Entry(prep_project_frame)
prep_project_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

# --- Database Selection ---
prep_database_frame = ttk.Frame(prep_data_frame)
prep_database_frame.pack(pady=5, padx=10, fill=tk.X)
prep_database_frame.columnconfigure(1, weight=1)
ttk.Button(prep_database_frame, text="Select Database File (.csv)",
           command=select_prep_database_file).grid(row=0, column=0,
                                                   padx=5, pady=5,
                                                   sticky=tk.W)
prep_database_entry = ttk.Entry(prep_database_frame)
prep_database_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

# --- Output Locations ---
prep_results_frame = ttk.Frame(prep_data_frame)
prep_results_frame.pack(pady=5, padx=10, fill=tk.X)
prep_results_frame.columnconfigure(1, weight=1)
# dem folder
ttk.Button(prep_results_frame, text="Select DEM Folder",
           command=select_prep_dem_dir).grid(row=0, column=0,
                                             padx=5, pady=5,
                                             sticky=tk.W)
prep_dem_entry = ttk.Entry(prep_results_frame)
prep_dem_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
# hydrography folder
ttk.Button(prep_results_frame, text="Select Hydrography Folder",
           command=select_prep_strm_dir).grid(row=1, column=0,
                                              padx=5, pady=5,
                                              sticky=tk.W)
prep_strm_entry = ttk.Entry(prep_results_frame)
prep_strm_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
# results folder
ttk.Button(prep_results_frame, text="Select Results Folder",
           command=select_prep_results_dir).grid(row=2, column=0,
                                                 padx=5, pady=5,
                                                 sticky=tk.W)
prep_results_entry = ttk.Entry(prep_results_frame)
prep_results_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
# RathCelon .json input
ttk.Button(prep_results_frame, text="RathCelon Input File (.json)",
           command=select_prep_json_file).grid(row=3, column=0,
                                               padx=5, pady=5,
                                               sticky=tk.W)
prep_json_entry = ttk.Entry(prep_results_frame)
prep_json_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.EW)


# --- Hydraulics and Hydrology Settings ---
prep_hydro_frame = ttk.Frame(prep_data_frame)
prep_hydro_frame.pack(pady=5, padx=10, fill=tk.X)
prep_hydro_frame.columnconfigure(1, weight=1)
ttk.Label(prep_hydro_frame, text="Hydrography Data Source:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
prep_hydro_var = tk.StringVar(value="NHDPlus")
prep_hydro_dropdown = ttk.Combobox(prep_hydro_frame, textvariable=prep_hydro_var, state="readonly",
                                   values=("NHDPlus", "GEOGLOWS"))
prep_hydro_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
ttk.Label(prep_hydro_frame, text="Preferred DEM Resolution:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
prep_dd_var = tk.StringVar(value="1 m")
prep_dd_dropdown = ttk.Combobox(prep_hydro_frame, textvariable=prep_dd_var, state="readonly",
                                values=("1 m", "1/9 arc-second (~3 m)", "1/3 arc-second (~10 m)"))
prep_dd_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
ttk.Label(prep_hydro_frame, text="Hydrology Data Source:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
prep_logy_var = tk.StringVar(value="National Water Model")
prep_logy_dropdown = ttk.Combobox(prep_hydro_frame, textvariable=prep_logy_var, state="readonly",
                                  values=("National Water Model", "GEOGLOWS"))
prep_logy_dropdown.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
ttk.Label(prep_hydro_frame, text="Baseflow Estimation:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
prep_baseflow_var = tk.StringVar(value="WSE and LiDAR Date")
prep_baseflow_dropdown = ttk.Combobox(prep_hydro_frame, textvariable=prep_baseflow_var, state="readonly", values=(
    "WSE and LiDAR Date", "WSE and Median Daily Flow", "2-yr Flow and Bank Estimation"))
prep_baseflow_dropdown.grid(row=3, column=1, padx=5, pady=5, sticky=tk.EW)

# --- Run function button (Step 1) ---
prep_run_button = ttk.Button(prep_data_frame, text="1. Prepare Data & Create Input File", command=start_prep_thread,
                             style="Accent.TButton")
prep_run_button.pack(pady=10, padx=10, fill=tk.X, ipady=5)

# --- Frame for Step 2: Run Rathcelon ---
run_rathcelon_frame = ttk.LabelFrame(prep_tab, text="Step 2: Run Rathcelon Processing")
run_rathcelon_frame.pack(pady=10, padx=10, fill=tk.X)
run_rathcelon_frame.columnconfigure(1, weight=1)

ttk.Button(run_rathcelon_frame, text="Select Input File (.json)", command=select_rathcelon_json_file).grid(row=0,
                                                                                                           column=0,
                                                                                                           padx=5,
                                                                                                           pady=5,
                                                                                                           sticky=tk.W)
rathcelon_json_entry = ttk.Entry(run_rathcelon_frame)
rathcelon_json_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

rathcelon_run_button = ttk.Button(run_rathcelon_frame, text="2. Run RathCelon",
                                  command=start_rathcelon_run_thread,
                                  style="Accent.TButton")
rathcelon_run_button.grid(row=1, column=0, columnspan=2, padx=5, pady=10, sticky=tk.EW, ipady=5)

"""
===================================================================
                 --- GUI: ANALYSIS TAB ---
===================================================================
"""

# --- Analysis: Database and Results Paths ---
analysis_path_frame = ttk.LabelFrame(analysis_tab, text="Database and Results Paths")
analysis_path_frame.pack(pady=10, padx=10, fill=tk.X, side=tk.TOP)
analysis_path_frame.columnconfigure(1, weight=1)
ttk.Button(analysis_path_frame, text="Select Database File (.csv)", command=select_analysis_csv_file).grid(row=0,
                                                                                                           column=0,
                                                                                                           padx=5,
                                                                                                           pady=5,
                                                                                                           sticky=tk.W)
analysis_database_entry = ttk.Entry(analysis_path_frame)
analysis_database_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
ttk.Button(analysis_path_frame, text="Select Results Folder", command=select_analysis_results_dir).grid(row=1, column=0,
                                                                                                        padx=5, pady=5,
                                                                                                        sticky=tk.W)
analysis_results_entry = ttk.Entry(analysis_path_frame)
analysis_results_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

# --- Analysis: Settings ---
analysis_settings_frame = ttk.LabelFrame(analysis_tab, text="Analysis Settings")
analysis_settings_frame.pack(pady=10, padx=10, fill=tk.X, side=tk.TOP)
analysis_settings_frame.columnconfigure(1, weight=1)
ttk.Label(analysis_settings_frame, text="Hydrologic Data Source:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
analysis_model_var = tk.StringVar(value="National Water Model")
analysis_model_dropdown = ttk.Combobox(analysis_settings_frame, textvariable=analysis_model_var, state="readonly",
                                       values=("USGS", "GEOGLOWS", "National Water Model"))
analysis_model_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
analysis_estimate_dam_var = tk.BooleanVar(value=True)
analysis_checkbox = ttk.Checkbutton(analysis_settings_frame, text="Estimate Dam Height",
                                    variable=analysis_estimate_dam_var)
analysis_checkbox.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
ttk.Label(analysis_settings_frame, text="Dam(s) to Analyze:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
analysis_dam_dropdown = ttk.Combobox(analysis_settings_frame, state="readonly")
analysis_dam_dropdown.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)

# --- Analysis: Run Buttons ---
analysis_button_frame = ttk.Frame(analysis_tab)
analysis_button_frame.pack(pady=10, fill=tk.X, padx=10, side=tk.TOP)
analysis_button_frame.columnconfigure(0, weight=1)
analysis_button_frame.columnconfigure(1, weight=1)
analysis_run_button = ttk.Button(analysis_button_frame, text="4. Analyze & Save Dam Data", command=start_analysis_processing)
analysis_run_button.grid(row=0, column=0, padx=5, ipady=5, sticky=tk.EW)
analysis_summary_button = ttk.Button(analysis_button_frame, text="Generate Bar Chart", # NEW BUTTON
                                     command=start_summary_plotting)
analysis_summary_button.grid(row=0, column=1, padx=5, ipady=5, sticky=tk.EW)

# --- Analysis: Figure Display Options ---
analysis_figure_frame = ttk.LabelFrame(analysis_tab, text="Select Figures to Display (for Single Dam Analysis)")
analysis_figure_frame.pack(pady=10, padx=10, fill=tk.X, side=tk.TOP)
analysis_figure_frame.columnconfigure(0, weight=1)
analysis_figure_frame.columnconfigure(1, weight=1)
analysis_display_cross_section = tk.BooleanVar(value=False)
ttk.Checkbutton(analysis_figure_frame, text="Cross-Sections", variable=analysis_display_cross_section).grid(row=0,
                                                                                                            column=0,
                                                                                                            padx=5,
                                                                                                            pady=2,
                                                                                                            sticky=tk.W)
analysis_display_rating_curves = tk.BooleanVar(value=False)
ttk.Checkbutton(analysis_figure_frame, text="Rating Curves", variable=analysis_display_rating_curves).grid(row=1,
                                                                                                           column=0,
                                                                                                           padx=5,
                                                                                                           pady=2,
                                                                                                           sticky=tk.W)
analysis_display_map = tk.BooleanVar(value=False)
ttk.Checkbutton(analysis_figure_frame, text="Dam Location", variable=analysis_display_map).grid(row=2, column=0, padx=5,
                                                                                                pady=2, sticky=tk.W)
analysis_display_wsp = tk.BooleanVar(value=False)
ttk.Checkbutton(analysis_figure_frame, text="Water Surface Profile", variable=analysis_display_wsp).grid(row=0,
                                                                                                         column=1,
                                                                                                         padx=5, pady=2,
                                                                                                         sticky=tk.W)
analysis_display_fdc = tk.BooleanVar(value=False)
ttk.Checkbutton(analysis_figure_frame, text="Flow-Duration Curve", variable=analysis_display_fdc).grid(row=1, column=1,
                                                                                                       padx=5, pady=2,
                                                                                                       sticky=tk.W)

# --- Analysis: NEW Display Button ---
analysis_display_button_frame = ttk.Frame(analysis_tab)
analysis_display_button_frame.pack(pady=10, fill=tk.X, padx=10, side=tk.TOP)
analysis_display_button = ttk.Button(analysis_display_button_frame, text="5. Generate & Display Dam Figures",
                                     command=start_display_dam_figures_thread, style="Accent.TButton")
analysis_display_button.pack(fill=tk.X, ipady=5)


# --- Analysis: NEW Figure Viewer Frame ---
analysis_figure_viewer_frame = ttk.LabelFrame(analysis_tab, text="Figure Viewer")
# This frame is packed at the end, but its content is filled by functions

# This frame will hold the controls (buttons, label) - PACKED FIRST
analysis_figure_controls_frame = ttk.Frame(analysis_figure_viewer_frame)
analysis_figure_controls_frame.pack(fill="x", pady=5, side=tk.TOP) # Explicitly pack at the top

prev_button = ttk.Button(analysis_figure_controls_frame, text="< Previous", command=on_prev_figure)
prev_button.pack(side=tk.LEFT, padx=10)

next_button = ttk.Button(analysis_figure_controls_frame, text="Next >", command=on_next_figure)
next_button.pack(side=tk.RIGHT, padx=10)

analysis_figure_label_var = tk.StringVar(value="No figure loaded.")
analysis_figure_label = ttk.Label(analysis_figure_controls_frame, textvariable=analysis_figure_label_var,
                                  anchor=tk.CENTER)
analysis_figure_label.pack(side=tk.LEFT, fill="x", expand=True)

# This frame will hold the matplotlib canvas - PACKED SECOND
analysis_figure_canvas_frame = ttk.Frame(analysis_figure_viewer_frame)
analysis_figure_canvas_frame.pack(fill="both", expand=True)

# Pack the main viewer frame (it will be un-packed by clear_figure_carousel)
analysis_figure_viewer_frame.pack(fill="both", expand=True, padx=10, pady=10)
# Hide the figure viewer frame initially
analysis_figure_viewer_frame.pack_forget()

"""
===================================================================
                        --- STATUS BAR ---
===================================================================
"""
status_var = tk.StringVar()
status_var.set("Ready. Please select a Project Folder in the 'Preparation' tab to begin.")
status_label = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
status_label.pack(side=tk.BOTTOM, fill=tk.X, ipady=2)

root.mainloop()