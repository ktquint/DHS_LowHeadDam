# DHS_LowHeadDam Processor

This project analyzes the downstream flow conditions at low-head dams using location and width data. It integrates functionality from the [RathCelon](https://github.com/jlgutenson/rathcelon) and [ARC](https://github.com/MikeFHS/automated-rating-curve) packages.

This guide provides instructions for setting up the required environment using Conda, which is the recommended method for ensuring all dependencies (especially complex GIS packages) are installed correctly on Windows, macOS, and Linux.

---

## 1. Setup Instructions (Do this only once)

1.  **Install Miniconda:**
    * Go to the [Miniconda documentation](https://docs.conda.io/en/latest/miniconda.html) and download the installer for your operating system (Windows or macOS).
    * Run the installer and accept all the default settings.

2.  **Download the Project Code:**
    * Download this repository as a ZIP file (or use `git clone`).
    * Unzip the folder. It should already contain an `environment.yaml` file.

3.  **Create the Conda Environment:**
    * **On Windows:** Open the **"Anaconda Prompt"** (or "Miniconda Prompt") from your Start Menu.
    * **On macOS:** Open the **"Terminal"** application.
    * Navigate to the project folder you downloaded. Use `cd` (change directory). For example:
        ```bash
        cd C:\Users\YourName\Downloads\DHS_LowHeadDam-main
        ```
        *(Tip: You can drag and drop the folder onto the terminal window to get its path).*
    * Run the following command to create the environment. This will read the `environment.yaml` file and install all packages. This step will take 5-10 minutes.
        ```bash
        conda env create -f environment.yaml
        ```

---

## 2. How to Run the Program (Do this every time)

1.  **Activate the Environment:**
    * Open your terminal (Anaconda Prompt on Windows, Terminal on macOS).
    * Type the following command to "activate" the environment and all its packages:
        ```bash
        conda activate lhd-environment
        ```
    * You will see `(lhd-environment)` appear at the beginning of your command prompt, showing it's active.

2.  **Navigate to the `src` Directory:**
    * From the same terminal, navigate *inside* the project's `src` folder. This is critical for the script to find its helper files.
        ```bash
        # If you are in the main project folder:
        cd lhd_processor
        ```

3.  **Run the Processor:**
    * Now, simply run the python script to launch the GUI:
        ```bash
        python __main__.py
        ```

4.  **Deactivate (When you are done):**
    * When you close the program, you can "deactivate" the environment by running:
        ```bash
        conda deactivate
        ```