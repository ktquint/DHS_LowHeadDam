# Low-Head Dam Processor

This project analyzes the downstream flow conditions at low-head dams using location and width data. It integrates functionality from the [RathCelon](https://github.com/jlgutenson/rathcelon) and [ARC](https://github.com/MikeFHS/automated-rating-curve) packages.

The application is structured as a Python package (`lhd_processor`) and is run directly from your terminal.

This guide provides instructions for setting up the required environment using Conda, which is the recommended method for ensuring all dependencies (especially complex GIS packages) are installed correctly on Windows, macOS, and Linux.

---

## 1. Setup Instructions (Do this only once)

1.  **Install Miniconda:**
    * Go to the [Miniconda documentation](https://docs.conda.io/en/latest/miniconda.html) and download the installer for your operating system (Windows or macOS).
    * Run the installer and accept all the default settings.

2.  **Download the Project Code:**
    * Download this repository as a ZIP file (or use `git clone`).
    * Unzip the folder. It already contains an `environment.yaml` file.

3.  **Create the Conda Environment:**
    * **Open Terminal/Anaconda Prompt** and navigate to the **root project folder** that contains `environment.yaml`.
    * Run the following command to create the environment (named `lhd-environment`). This step will take 5-10 minutes.
        ```bash
        conda env create -f environment.yaml
        ```

---

## 2. Project Directory Organization (Do this once per project)

To use the LHD Processor effectively, it is recommended to set up a dedicated folder for your project. The application is designed to auto-populate paths based on this structure.

### Recommended Structure
Create a folder (e.g., `My_Dam_Project`) and place your input CSV file inside it. When you run the application and select this "Project Folder" in the GUI, the tool will automatically create the standard subdirectories for you.

```text
My_Dam_Project/
│
├── my_dams_database.csv   <-- Your required input file
│
├── LHD_Results/           <-- Auto-generated: Stores DEMs, analysis outputs, and figures
│   ├── [Dam_ID]/
│   │   ├── DEM/
│   │   ├── FIGS/
│   │   └── ...
│
├── LHD_STRMs/             <-- Auto-generated: Stores downloaded flowline data (NHD/GEOGLOWS)
│
└── my_dams_database.json  <-- Auto-generated: Input file created for the RathCelon step
```

---

## 3. How to Run the Program (Do this every time)

1.  **Activate the Environment:**
    * Open your terminal (Anaconda Prompt on Windows, Terminal on macOS).
    * Type the following command:
        ```bash
        conda activate lhd-environment
        ```
    * The prompt will show `(lhd-environment)` when active.

2.  **Run the Processor (as a Module):**
    * Navigate to the **root project directory** where the `lhd_processor` folder is located.
    * Use the module execution command (`-m`) to launch the GUI:
        ```bash
        python -m lhd_processor
        ```

3.  **Deactivate (When you are done):**
    * When you close the program, you can "deactivate" the environment by running:
        ```bash
        conda deactivate
        ```