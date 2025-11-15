import os
from dam import Dam
import pandas as pd
import xarray as xr
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import create_json as cj
import threading  # Import threading

# *** ADD THIS IMPORT ***
# This assumes 'rathcelon' is installed in your environment
try:
    from rathcelon.classes import Dam as RathcelonDam
except ImportError:
    messagebox.showerror("Import Error",
                         "Could not find 'rathcelon' package. \nPlease ensure it is installed in your environment.")
    # You could also add sys.exit() here if you want


# --- Helper Functions for GUI ---

def select_project_dir():
    project_path = filedialog.askdirectory()
    if not project_path:
        return  # User canceled

    project_entry.delete(0, tk.END)
    project_entry.insert(0, project_path)

    try:
        # ------------------------ FIND THE LHD DATABASE --------------------------- #
        excel_files = [f for f in os.listdir(project_path) if f.endswith('.xlsx')]
        if not excel_files:
            messagebox.showwarning("No Database", f"No .xlsx database file found in:\n{project_path}")
            return

        database_path = os.path.join(project_path, excel_files[0])
        database_entry.delete(0, tk.END)
        database_entry.insert(0, database_path)

        # ---------------- AUTO-POPULATE OTHER PATHS ------------------- #

        # This is the base "Results" path
        results_path = os.path.join(project_path, "LHD_Results")

        # *** THIS IS THE FIX ***
        # The DEM folder expected by download_dem is the main Results folder,
        # not the {ID}/DEM subfolder.
        dem_path = results_path
        dem_entry.delete(0, tk.END)
        dem_entry.insert(0, dem_path)

        strm_path = os.path.join(project_path, "LHD_STRMs")
        strm_entry.delete(0, tk.END)
        strm_entry.insert(0, strm_path)

        # This entry also points to the base Results path
        results_entry.delete(0, tk.END)
        results_entry.insert(0, results_path)

        json_path = os.path.join(project_path, "rathcelon_input.json")
        json_entry.delete(0, tk.END)
        json_entry.insert(0, json_path)

        csv_path = os.path.splitext(database_path)[0] + '.csv'
        csv_entry.delete(0, tk.END)
        csv_entry.insert(0, csv_path)

        status_var.set("Project paths loaded.")

    except IndexError:
        messagebox.showerror("Error", f"No .xlsx database file found in:\n{project_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load project paths: {e}")


def select_database_file():
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    if file_path:
        database_entry.delete(0, tk.END)
        database_entry.insert(0, file_path)


def select_dem_dir():
    dem_path = filedialog.askdirectory()
    if dem_path:
        dem_entry.delete(0, tk.END)
        dem_entry.insert(0, dem_path)


def select_strm_dir():
    strm_path = filedialog.askdirectory()
    if strm_path:
        strm_entry.delete(0, tk.END)
        strm_entry.insert(0, strm_path)


def select_results_dir():
    results_path = filedialog.askdirectory()
    if results_path:
        results_entry.delete(0, tk.END)
        results_entry.insert(0, results_path)


def select_json_file():
    json_path = filedialog.asksaveasfilename(filetypes=[("JSON files", "*.json")], defaultextension=".json")
    if json_path:
        json_entry.delete(0, tk.END)
        json_entry.insert(0, json_path)


def select_csv_file():
    csv_path = filedialog.asksaveasfilename(filetypes=[("CSV files", "*.csv")], defaultextension=".csv")
    if csv_path:
        csv_entry.delete(0, tk.END)
        csv_entry.insert(0, csv_path)


# --- THE MEAT OF THIS PROGRAM --- #

def create_input_file():
    """
    This function now runs in a separate thread.
    It contains all the core logic, error handling, and status updates.
    """
    try:
        # --- 1. Get all values from GUI ---
        lhd_xlsx = database_entry.get()
        hydrography = hydro_var.get()
        dem_resolution = dd_var.get()
        hydrology = logy_var.get()

        # This now correctly gets the base results path (e.g., "...\LHD_Results")
        dem_folder = dem_entry.get()

        strm_folder = strm_entry.get()
        results_folder = results_entry.get()
        lhd_csv = csv_entry.get()

        # --- 2. Validate inputs ---
        if not os.path.exists(lhd_xlsx):
            messagebox.showerror("Error", f"Database file not found:\n{lhd_xlsx}")
            return
        if not all(
                [lhd_xlsx, hydrography, dem_resolution, hydrology, dem_folder, strm_folder, results_folder, lhd_csv]):
            messagebox.showwarning("Missing Info", "Please fill out all path and setting fields.")
            return

        status_var.set("Inputs validated. Starting data prep...")

        # --- 3. Create directories ---
        os.makedirs(dem_folder, exist_ok=True)
        os.makedirs(strm_folder, exist_ok=True)
        os.makedirs(results_folder, exist_ok=True)

        lhd_df = pd.read_excel(lhd_xlsx)

        # --- 4. Load NWM data once ---
        nwm_ds = None
        if hydrology == 'National Water Model':
            status_var.set("Loading NWM dataset...")
            try:
                s3_path = 's3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr'
                nwm_ds = xr.open_zarr(s3_path, consolidated=True, storage_options={"anon": True})
                status_var.set("NWM dataset loaded.")
            except Exception as e:
                print(f"Could not open NWM zarr dataset. NWM streamflow will not be processed. Error: {e}")
                status_var.set("Error: Could not load NWM dataset.")
                nwm_ds = None

        # --- 5. Main Processing Loop ---
        total_dams = len(lhd_df)
        processed_dams_count = 0
        final_df = lhd_df.copy()

        for i, row in lhd_df.iterrows():
            dam_id = row.get("ID", f"Row_{i + 2}")

            try:
                status_var.set(f"Prep: Dam {dam_id} ({i + 1} of {total_dams})...")

                dam = Dam(**row.to_dict())
                dam.assign_hydrology(hydrology)
                dam.assign_hydrography(hydrography)

                status_var.set(f"Dam {dam_id}: Assigning flowlines...")
                dam.assign_flowlines(strm_folder, "")

                status_var.set(f"Dam {dam_id}: Assigning DEM...")
                # This call now works, because dem_folder is the base results path.
                # dam.assign_dem will call download_dem, which will create:
                # ".../LHD_Results/{dam_id}/DEM"
                dam.assign_dem(dem_folder, dem_resolution)

                if dam.dem_1m is None and dam.dem_3m is None and dam.dem_10m is None:
                    print(f"Skipping Dam No. {dam_id}: DEM assignment failed.")
                    status_var.set(f"Dam {dam_id}: DEM failed. Skipping.")
                    continue

                    # This function just assigns the string, it's fine.
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
                        dam.set_dem_baseflow()

                        status_var.set(f"Dam {dam_id}: Estimating fatal flows...")
                        dam.set_fatal_flows()

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
                print(f"CRITICAL ERROR processing Dam No. {dam_id}: {e}")
                print(f"Skipping this dam and moving to the next one.")
                print(f"---" * 20)
                status_var.set(f"Error on Dam {dam_id}: {e}. Skipping.")
                continue

                # --- 6. Final Output Generation (CSV/Excel) ---
        if processed_dams_count > 0:
            status_var.set("Saving updated database files...")

            final_df.to_csv(lhd_csv, index=False)
            final_df.to_excel(lhd_xlsx, index=False)

            status_var.set("Creating rathcelon_input.json...")
            json_loc = json_entry.get()

            # Call create_json, which writes the file AND returns the dam dicts
            dam_dictionaries = cj.rathcelon_input(lhd_csv, json_loc, hydrography, hydrology)

            status_var.set(f"Data prep complete. Now running RathCelon analysis...")

            # --- 7. RUN RATHCELON PROCESS ---
            if not dam_dictionaries:
                status_var.set("No dams to process in RathCelon.")
                return

            rathcelon_success_count = 0
            for i, dam_dict in enumerate(dam_dictionaries):
                dam_name = dam_dict.get('name', f"Dam {i + 1}")
                try:
                    status_var.set(f"RathCelon: Processing Dam {dam_name} ({i + 1} of {len(dam_dictionaries)})...")

                    # This creates an instance of the rathcelon.classes.Dam
                    # *** NOTE: This requires 'dem_dir' in the dict to be the *base* results folder
                    # Let's fix the dam_dict before passing it

                    # The 'dem_dir' rathcelon expects is the one *inside* the {ID} folder
                    # Your create_json.py already creates this path correctly!
                    # "dem_dir" : dem_dir (which was row['dem_1m'] etc.)
                    # which IS "...\LHD_Results/{ID}/DEM"

                    # Wait, let's re-read the rathcelon files.
                    # rathcelon/main.py -> process_dam(dam_dict)
                    # DEM_Folder = dam_dict['dem_dir']
                    # DEM_List = os.listdir(DEM_Folder)
                    # This means rathcelon *does* expect the final ".../{ID}/DEM" folder.

                    # My previous assumption was wrong. Your `create_json.py` is
                    # correctly creating the "dem_dir" key that rathcelon expects.

                    # The problem is that `create_json.py` gets the `dem_dir` path
                    # from the *database row* (e.g., `row['dem_1m']`), which is
                    # populated by *your* `dam.assign_dem` function.

                    # This means `dam.assign_dem` MUST be given the correct
                    # base path so it can create the *correct* final path and
                    # store it in the `dam.dem_1m` attribute, which is then
                    # saved to the CSV, which is *then* read by `create_json.py`.

                    # My change to `select_project_dir` was correct.
                    # The entire flow is:
                    # 1. GUI `dem_entry` gets "...\LHD_Results"
                    # 2. `create_input_file` gets this path as `dem_folder`.
                    # 3. `dam.assign_dem(dem_folder, ...)` is called.
                    # 4. `download_dem` (called by `assign_dem`) creates `dem_subdir = "...\LHD_Results\{ID}\DEM"`
                    # 5. `assign_dem` saves this `dem_subdir` to `dam.dem_1m`.
                    # 6. This `dam` object (with `dam.dem_1m = "...\LHD_Results\{ID}\DEM"`) is saved to the CSV.
                    # 7. `cj.rathcelon_input` reads this CSV, gets the "...\LHD_Results\{ID}\DEM" path,
                    #    and puts it into the `dam_dict['dem_dir']`.
                    # 8. This `dam_dict` is passed to `RathcelonDam(**dam_dict)`.
                    # 9. `RathcelonDam`'s `process_dam` function gets `DEM_Folder = dam_dict['dem_dir']`,
                    #    which is "...\LHD_Results\{ID}\DEM".
                    # 10. It then does `DEM_List = os.listdir(DEM_Folder)`, which lists the .tif files.

                    # This logic is sound. My change was correct.

                    dam_i = RathcelonDam(**dam_dict)
                    dam_i.process_dam()

                    rathcelon_success_count += 1
                except Exception as e:
                    print(f"RathCelon failed for dam {dam_name}: {e}")
                    status_var.set(f"RathCelon Error on Dam {dam_name}. Skipping.")
                    # Continue to the next dam

            status_var.set(f"Process complete. {rathcelon_success_count} dams processed by RathCelon.")
            messagebox.showinfo("Success", f"Process complete. {rathcelon_success_count} dams processed by RathCelon.")

        else:
            status_var.set("No dams were pre-processed successfully.")
            messagebox.showwarning("Process Finished", "No new dam data was pre-processed successfully.")

    except Exception as e:
        status_var.set(f"Fatal error: {e}")
        messagebox.showerror("Fatal Error", f"The process failed:\n{e}")
    finally:
        # --- 8. Re-enable button ---
        run_button.config(state=tk.NORMAL)


def start_processing_thread():
    """
    This function is the new command for the run_button.
    It disables the button and starts the `create_input_file` function in a new thread.
    """
    run_button.config(state=tk.DISABLED)
    status_var.set("Starting process...")
    threading.Thread(target=create_input_file, daemon=True).start()


# --- GUI setup ---
root = tk.Tk()
root.title("RathCelon Input File Generator & Runner")
root.geometry("600x750")

# --- Project Directory Selection ---
project_frame = ttk.LabelFrame(root, text="Project Folder")
project_frame.pack(pady=10, padx=10, fill=tk.X)
project_frame.columnconfigure(1, weight=1)

ttk.Button(project_frame, text="Select Project Folder", command=select_project_dir).grid(row=0, column=0, padx=5,
                                                                                         pady=5, sticky=tk.W)
project_entry = ttk.Entry(project_frame)
project_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

# --- Database Selection ---
database_frame = ttk.LabelFrame(root, text="Input Database")
database_frame.pack(pady=5, padx=10, fill=tk.X)
database_frame.columnconfigure(1, weight=1)

ttk.Button(database_frame, text="Select Database File (.xlsx)", command=select_database_file).grid(row=0, column=0,
                                                                                                   padx=5, pady=5,
                                                                                                   sticky=tk.W)
database_entry = ttk.Entry(database_frame)
database_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

# --- Output Locations ---
results_frame = ttk.LabelFrame(root, text="Output Locations")
results_frame.pack(pady=10, padx=10, fill=tk.X)
results_frame.columnconfigure(1, weight=1)

ttk.Button(results_frame, text="Select DEM Folder", command=select_dem_dir).grid(row=0, column=0, padx=5, pady=5,
                                                                                 sticky=tk.W)
dem_entry = ttk.Entry(results_frame)
dem_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

ttk.Button(results_frame, text="Select Hydrography Folder", command=select_strm_dir).grid(row=1, column=0, padx=5,
                                                                                          pady=5, sticky=tk.W)
strm_entry = ttk.Entry(results_frame)
strm_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

ttk.Button(results_frame, text="Select Results Folder", command=select_results_dir).grid(row=2, column=0, padx=5,
                                                                                         pady=5, sticky=tk.W)
results_entry = ttk.Entry(results_frame)
results_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)

ttk.Button(results_frame, text="RathCelon Input File (.json)", command=select_json_file).grid(row=3, column=0, padx=5,
                                                                                              pady=5, sticky=tk.W)
json_entry = ttk.Entry(results_frame)
json_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.EW)

ttk.Button(results_frame, text="RathCelon Database File (.csv)", command=select_csv_file).grid(row=4, column=0, padx=5,
                                                                                               pady=5, sticky=tk.W)
csv_entry = ttk.Entry(results_frame)
csv_entry.grid(row=4, column=1, padx=5, pady=5, sticky=tk.EW)

# --- Hydraulics and Hydrology Settings ---
hydro_frame = ttk.LabelFrame(root, text="Hydrologic Data Information")
hydro_frame.pack(pady=10, padx=10, fill=tk.X)
hydro_frame.columnconfigure(1, weight=1)

ttk.Label(hydro_frame, text="Hydrography Data Source:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
hydro_var = tk.StringVar(value="NHDPlus")
hydro_dropdown = ttk.Combobox(hydro_frame, textvariable=hydro_var, state="readonly", values=("NHDPlus", "GEOGLOWS"))
hydro_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)

ttk.Label(hydro_frame, text="Preferred DEM Resolution:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
dd_var = tk.StringVar(value="1 m")
dd_dropdown = ttk.Combobox(hydro_frame, textvariable=dd_var, state="readonly",
                           values=("1 m", "1/9 arc-second (~3 m)", "1/3 arc-second (~10 m)"))
dd_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

ttk.Label(hydro_frame, text="Hydrology Data Source:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
logy_var = tk.StringVar(value="National Water Model")
logy_dropdown = ttk.Combobox(hydro_frame, textvariable=logy_var, state="readonly",
                             values=("National Water Model", "GEOGLOWS"))
logy_dropdown.grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)

ttk.Label(hydro_frame, text="Baseflow Estimation:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
baseflow_var = tk.StringVar(value="WSE and LiDAR Date")
baseflow_dropdown = ttk.Combobox(hydro_frame, textvariable=baseflow_var, state="readonly", values=(
"WSE and LiDAR Date", "WSE and Median Daily Flow", "2-yr Flow and Bank Estimation"))
baseflow_dropdown.grid(row=3, column=1, padx=5, pady=5, sticky=tk.EW)

# --- Run function button ---
run_button = ttk.Button(root, text="Prepare Data and Run RathCelon", command=start_processing_thread,
                        style="Accent.TButton")
run_button.pack(pady=10, padx=10, fill=tk.X, ipady=5)

# --- Status Bar ---
status_var = tk.StringVar()
status_var.set("Ready.")
status_label = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
status_label.pack(side=tk.BOTTOM, fill=tk.X)

# --- Add a style for the main button ---
style = ttk.Style()
style.configure("Accent.TButton", font=("Arial", 10, "bold"))

root.mainloop()
