import pandas as pd
import geopandas as gpd
import os
import requests

# Source of data (initial excel file) - Remove later when df_slopes is available; use this for practice runs
file_path = "Low head Dam Info - Copy for python.xlsx"
df_slopes = pd.read_excel(file_path, usecols=['latitude', 'longitude', 'ID', 'LINKNO', 'gpkg'])

# Downloads the gpkg file later in the code
def download_gpkg(url, local_path):
    response = requests.get(url)
    with open(local_path, 'wb') as f:
        f.write(response.content)

def gpkg_download(df_slopes):
    # Place where gpkgs are downloaded
    download_dir = 'all_downloaded_gpkgs'
    os.makedirs(download_dir, exist_ok=True)

    #List of linknos, will just put the list into lhd
    linknos = pd.read_csv('list_of_linkno.csv')

    # Ensure the 'gpkg' column is of type object
    df_slopes['gpkg'] = df_slopes['gpkg'].astype(object)

    # looks through rows of the initial dataframe
    for row in df_slopes.itertuples():
        linkno = row.LINKNO # gets linkno value of the row

        if isinstance(linkno, int): # Checks if linkno is int
            found = False
            for column in linknos.columns: # checks thru all columns of linkno values csv
                local_gpkg_path = os.path.join(download_dir, f"{column}.gpkg") # path to store gpkg
                gpkg_url = f"http://geoglows-v2.s3-us-west-2.amazonaws.com/streams/{column}.gpkg" # where gpkg is downloaded from
                if linkno in linknos[column].values: # if linkno is in the column of the gpkg
                    if not os.path.exists(local_gpkg_path): # if wasn't already downloaded
                        download_gpkg(gpkg_url, local_gpkg_path) # downloads
                        df_slopes.at[row.Index, 'gpkg'] = f"{column}.gpkg" # adds gpkg name to the excel file
                        found = True
                        break
                    elif os.path.exists(local_gpkg_path): # if linkno is already downloaded
                        df_slopes.at[row.Index, 'gpkg'] = f"{column}.gpkg" # just adds gpkg name, no download
                        found = True
                        break
            if not found:
                df_slopes.at[row.Index, 'gpkg'] = "" # adds nothing if not found

    df_slopes.to_excel('output_w_gpkgs.xlsx', index=False) # converts to excel file

gpkg_download(df_slopes) #Works when I tested it, please check files