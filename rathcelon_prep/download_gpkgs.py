import os
import requests
import pandas as pd
import pathlib


# Source of data (initial excel file) - Remove later when df_slopes is available; use this for practice runs
file_path = "../Low head Dam Info - Copy for python.xlsx"                                           # Excel with values
df_slopes = pd.read_excel(file_path, usecols=['latitude', 'longitude', 'ID', 'LINKNO', 'gpkg'])     # Dataframe of values

# Downloads the gpkg file later in the code
def download_gpkg(url, local_path):
    response = requests.get(url)
    with open(local_path, 'wb') as f:
        f.write(response.content)

def gpkg_download(df_slopes):
    gpkg_list = []                                  # List to store gpkg values (paths)
    download_dir = 'all_downloaded_gpkgs'           # Place where gpkgs are downloaded
    os.makedirs(download_dir, exist_ok=True)        # Makes the directory to store the gpkg files; if already exists, won't do anything

    #List of linknos for each gpkg created beforehand
    linknos = pd.read_csv('../list_of_linkno.csv')

    # Ensure the 'gpkg' column is of type object
    df_slopes['gpkg'] = df_slopes['gpkg'].astype(object)

    # looks through rows of the initial dataframe
    for row in df_slopes.itertuples():
        linkno = row.LINKNO         # gets linkno value of the row

        if isinstance(linkno, int): # Checks if linkno is int
            found = False           # Flag
            for column in linknos.columns:                                      # checks thru all columns of linkno values csv
                local_gpkg_path = os.path.join(download_dir, f"{column}.gpkg")  # path to store gpkg
                gpkg_url = f"http://geoglows-v2.s3-us-west-2.amazonaws.com/streams/{column}.gpkg" # where gpkg is downloaded from
                if linkno in linknos[column].values:                    # if linkno of dam is in the column of the linkno values of a gpkg
                    if not os.path.exists(local_gpkg_path):             # if wasn't already downloaded
                        download_gpkg(gpkg_url, local_gpkg_path)        # downloads
                        file_name = f"{column}.gpkg"                    # name of the gpkg file
                        current_directory = os.path.dirname(__file__)   # directory the current file is in
                        file_path = os.path.join(current_directory, download_dir, file_name)        # creates the path of the gpkg file
                        raw_file = repr(file_path)                      # turn path into raw string
                        path_to_gpkg = pathlib.PureWindowsPath(raw_file).as_posix()                 # convert into path with forward slashes
                        gpkg_list.append(path_to_gpkg)                  # adds to gpkg list
                        found = True
                        break
                    elif os.path.exists(local_gpkg_path):               # if linkno is already downloaded, skips download
                        file_name = f"{column}.gpkg"                    # name of the gpkg file
                        current_directory = os.path.dirname(__file__)   # directory the current file is in
                        file_path = os.path.join(current_directory, download_dir, file_name)        # creates the path of the gpkg file
                        raw_file = repr(file_path)                      # turn path into raw string
                        path_to_gpkg = pathlib.PureWindowsPath(raw_file).as_posix()                 # convert into path with forward slashes
                        gpkg_list.append(path_to_gpkg)                  # adds to gpkg list
                        found = True
                        break
            if not found:
                gpkg_list.append("")                                    # if doesn't find linkno in the gpkgs, adds nothing
        elif isinstance(linkno, str):
            gpkg_list.append("")                                        # adds empty space if says something like "No River ID" or non-number
    df_slopes['gpkg'] = pd.Series(gpkg_list)                            # adds the values of the list to the column of dataframe (faster processing)
    df_slopes.to_excel('output_w_gpkg_paths.xlsx', index=False)         # converts to excel file

gpkg_download(df_slopes) # runs function