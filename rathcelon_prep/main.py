import os
from dam import Dam
import pandas as pd
import xarray as xr
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import create_json as cj


def select_project_dir():
    project_path = filedialog.askdirectory()
    if project_path:
        project_entry.delete(0, tk.END)
        project_entry.insert(0, project_path)

        # ------------------------ FIND THE LHD DATABASE --------------------------- #
        database_path = [f for f in os.listdir(project_path) if f.endswith('.xlsx')][0]
        database_path = os.path.join(project_path, database_path)
        database_entry.delete(0, tk.END)
        database_entry.insert(0, database_path)

        # ---------------- DEM INFO ------------------- #
        dem_path = os.path.join(project_path, "LHD_DEMs")
        dem_entry.delete(0, tk.END)
        dem_entry.insert(0, dem_path)
        # -------------- HYDROGRAPHY INFO --------------- #
        strm_path = os.path.join(project_path, "LHD_STRMs")
        strm_entry.delete(0, tk.END)
        strm_entry.insert(0, strm_path)
        # ---------------- RESULTS INFO ---------------------- #
        results_path = os.path.join(project_path, "LHD_Results")
        results_entry.delete(0, tk.END)
        results_entry.insert(0, results_path)
        # ----- RATHCELON INPUT INFO ---------- #
        json_path = os.path.join(project_path, "rathcelon_input.json")
        json_entry.delete(0, tk.END)
        json_entry.insert(0, json_path)
        # ---- RATHCELON DATABASE ----- #
        csv_path = os.path.splitext(database_path)[0] + '.csv'
        csv_entry.delete(0, tk.END)
        csv_entry.insert(0, csv_path)

# - FUNCTIONS TO UPDATE OTHER FILES/DIRECTORIES - #
def select_database_file():
    file_path = filedialog.askopenfilename()
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
        dem_entry.delete(0, tk.END)
        dem_entry.insert(0, strm_path)


def select_results_dir():
    results_path = filedialog.askdirectory()
    if results_path:
        results_entry.delete(0, tk.END)
        results_entry.insert(0, results_path)


def select_json_file():
    json_path = filedialog.askopenfilename()
    if json_path:
        json_entry.delete(0, tk.END)
        json_entry.insert(0, json_path)


def select_csv_file():
    csv_path = filedialog.askopenfilename()
    if csv_path:
        csv_entry.delete(0, tk.END)
        csv_entry.insert(0, csv_path)

# --- THE MEAT OF THIS PROGRAM --- #
def create_input_file():
    """
    this is where the magic happens...
    """

    # this is the database I'm working with:
    lhd_xlsx = database_entry.get()
    hydrography = hydro_var.get()
    dem_resolution = dd_var.get()
    hydrology = logy_var.get()

    # these are where I'll store the DEMs and STRM.gpkg's
    dem_folder = dem_entry.get()
    os.makedirs(dem_folder, exist_ok=True)
    strm_folder = strm_entry.get()
    os.makedirs(strm_folder, exist_ok=True)
    # this is where I'll store the RathCelon output
    results_folder = results_entry.get()
    os.makedirs(results_folder, exist_ok=True)
    # we'll turn the finished data_frame into a csv with the name:
    lhd_csv = csv_entry.get()


    # convert your database to a data_frame
    lhd_df = pd.read_excel(lhd_xlsx)

    all_dams = []

    # if we're estimating the streamflows using the NWM, let's just download the data once...
    nwm_ds = None
    if hydrology == 'National Water Model':
        s3_path = 's3://noaa-nwm-retrospective-2-1-zarr-pds/chrtout.zarr'
        nwm_ds = xr.open_zarr(s3_path, consolidated=True, storage_options={"anon": True})

    # make all the stream reaches

    for _, row in lhd_df.iterrows():
        dam = Dam(**row.to_dict())
        # save the hydrology and hydrography info to the dam, so we don't have to pass them as fields later
        dam.assign_hydrology(hydrology)
        dam.assign_hydrography(hydrography)
        # find the right flowlines and save them to the strm folder
        dam.assign_flowlines(strm_folder)
        # find the DEM's around the dam, download them, and save their loc.
        dam.assign_dem(dem_folder, dem_resolution)
        # save the results folder loc.
        dam.assign_output(results_folder)
        # create a stream reach obj. we'll use to download fatal flows and DEM baseflow
        dam.create_reach(nwm_ds)
        # estimate the baseflow based on the hydrology option selected
        dam.est_dem_baseflow()
        dam.est_fatal_flows()
        all_dams.append(dam)
        print(f'Finished Dam No. {row["ID"]}')

    dam_df = pd.DataFrame([dam.__dict__ for dam in all_dams])
    dam_df.to_csv(lhd_csv, index=False) # save .csv copy for rathcelon
    dam_df.to_excel(lhd_xlsx, index=False) # update .xlsx
    print('Finished Downloading Data for All Requested Dams...')

    json_loc = json_entry.get()
    cj.rathcelon_input(lhd_csv, json_loc, hydrography, hydrology)
    # now type open the terminal in the rathcelon repository and type `rathcelon json ` + the input_loc path


# GUI setup
root = tk.Tk()
root.title("RathCelon Input File Generator")
root.geometry("600x800")

# Database label
input_label = tk.Label(root, text="Database Information", font=("Arial", 12, "bold"))
input_label.pack(pady=(15, 0))

# --- Project Directory Selection ---
project_frame = tk.Frame(root)
project_frame.pack(pady=10, padx=10, fill=tk.X)

tk.Button(project_frame, text="Select Folder", command=select_project_dir).pack(side=tk.LEFT)
project_entry = tk.Entry(project_frame)
project_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

# --- Database Selection ---
database_frame = tk.Frame(root)
database_frame.pack(pady=10, padx=10, fill=tk.X)

tk.Button(database_frame, text="Select Database File", command=select_database_file).pack(side=tk.LEFT)
database_entry = tk.Entry(database_frame)
database_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

# --- Results Directory Selection ---
section_label = tk.Label(root, text="Results Information", font=("Arial", 12, "bold"))
section_label.pack(pady=(15, 0))

## DEM directory
dem_frame = tk.Frame(root)
dem_frame.pack(pady=10, padx=10, fill=tk.X)

tk.Button(dem_frame, text="Select DEM Folder", command=select_dem_dir).pack(side=tk.LEFT)
dem_entry = tk.Entry(dem_frame)
dem_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

## STRM directory
strm_frame = tk.Frame(root)
strm_frame.pack(pady=10, padx=10, fill=tk.X)

tk.Button(strm_frame, text="Select Hydrography Folder", command=select_strm_dir).pack(side=tk.LEFT)
strm_entry = tk.Entry(strm_frame)
strm_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

## results directory
results_frame = tk.Frame(root)
results_frame.pack(pady=10, padx=10, fill=tk.X)

tk.Button(results_frame, text="Select Results Folder", command=select_results_dir).pack(side=tk.LEFT)
results_entry = tk.Entry(results_frame)
results_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

## input file location
json_frame = tk.Frame(root)
json_frame.pack(pady=10, padx=10, fill=tk.X)

tk.Button(json_frame, text="RathCelon Input File", command=select_json_file).pack(side=tk.LEFT)
json_entry = tk.Entry(json_frame)
json_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

## csv file location
csv_frame = tk.Frame(root)
csv_frame.pack(pady=10, padx=10, fill=tk.X)

tk.Button(csv_frame, text="RathCelon Database File", command=select_csv_file).pack(side=tk.LEFT)
csv_entry = tk.Entry(csv_frame)
csv_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)

# Hydraulics and Hydrology label
section_label = tk.Label(root, text="Hydrologic Data Information", font=("Arial", 12, "bold"))
section_label.pack(pady=(15, 0))

# --- HYDROGRAPHY Dropdown --- #
hydro_frame = tk.Frame(root)
hydro_frame.pack(pady=(5, 10), padx=10, fill=tk.X)

hydro_label = tk.Label(hydro_frame, text="Hydrography Data Source:")
hydro_label.pack(side=tk.LEFT, padx=(0, 10))

hydro_var = tk.StringVar()
hydro_dropdown = ttk.Combobox(hydro_frame, textvariable=hydro_var, state="readonly")
hydro_dropdown['values'] = ("NHDPlus", "GEOGLOWS")
hydro_dropdown.current(0)
hydro_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)

# ----- DEM DROPDOWN ----- #
dd_frame = tk.Frame(root)
dd_frame.pack(pady=(5, 10), padx=10, fill=tk.X)

dd_label = tk.Label(dd_frame, text="Preferred DEM Resolution: ")
dd_label.pack(side=tk.LEFT, padx=(0, 10))

dd_var = tk.StringVar()
dd_dropdown = ttk.Combobox(dd_frame, textvariable=dd_var, state="readonly")
dd_dropdown['values'] = ("1 m", "1/9 arc-second (~3 m)", "1/3 arc-second (~10 m)")
dd_dropdown.current(0)
dd_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)

# --- HYDROLOGY Dropdown --- #
logy_frame = tk.Frame(root)
logy_frame.pack(pady=(5, 10), padx=10, fill=tk.X)

logy_label = tk.Label(logy_frame, text="Hydrology Data Source:")
logy_label.pack(side=tk.LEFT, padx=(0, 10))

logy_var = tk.StringVar()
logy_dropdown = ttk.Combobox(logy_frame, textvariable=logy_var, state="readonly")
logy_dropdown['values'] = ("National Water Model", "GEOGLOWS")
logy_dropdown.current(0)
logy_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)

# --- EITHER KNOWN BASEFLOW OR Q50 WILL BE USED --- #
baseflow_frame = tk.Frame(root)
baseflow_frame.pack(pady=(5, 10), padx=10, fill=tk.X)

baseflow_label = tk.Label(baseflow_frame, text="Baseflow Estimation: ")
baseflow_label.pack(side=tk.LEFT, padx=(0, 10))

baseflow_var = tk.StringVar()
baseflow_dropdown = ttk.Combobox(baseflow_frame, textvariable=baseflow_var, state="readonly")
baseflow_dropdown['values'] = ("WSE and LiDAR Date", "WSE and Median Daily Flow", "2-yr Flow and Bank Estimation")
baseflow_dropdown.current(0)
baseflow_dropdown.pack(side=tk.LEFT, fill=tk.X, expand=True)


# --- Run function button ---
run_button = tk.Button(root, text="Create RathCelon Input File", command=create_input_file, height=2, width=20)
run_button.pack(pady=20)


root.mainloop()
