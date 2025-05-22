import os
import requests
import pandas as pd


def download_gpkg(gpkg_url, local_path):
    """
    Downloads the GeoGLows GPKGs locally.
    """
    if not os.path.exists(local_path):
        try:
            response = requests.get(gpkg_url)
            response.raise_for_status()  # Check if the request was successful
            with open(local_path, 'wb') as f:
                f.write(response.content)
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {gpkg_url}: {e}")
    else:
        print(f"Local file {local_path} already exists")


def find_col_name(df, value):
    for col in df.columns:
        if value in df[col].values:
            return col


def assign_flowlines(lhd_df, gpkg_dir):
    """
        Assigns flowlines to the LHD dataframe and downloads the necessary GPKGs.

        lhd_df_path: Path to the CSV file with lat, long, dem, etc. info.
        gpkg_dir: Directory where GPKGs will be downloaded.
        """
    lhd_df["flowline"] = None
    base_url = "https://geoglows-v2.s3-website-us-west-2.amazonaws.com/hydrography"
    gpkg_set = set()
    linkno_df = pd.read_csv("../list_of_linkno.csv")

    print("Starting Hydrography Download Process...")

    for index, row in lhd_df.iterrows():
        linkno = row['LINKNO'] # get the linkno for each dam
        gpkg_name = find_col_name(linkno_df, linkno) # find the gpkg with the linkno
        if gpkg_name:
             gpkg_path = f"{gpkg_dir}/{gpkg_name}.gpkg"
             lhd_df.at[index, 'flowline'] = str(gpkg_path)  # save the gpkg path for each dam
             gpkg_set.add(gpkg_path)
        else:
             print(f"LINKNO {linkno} not found in linkno_df")

    # Save updated DataFrame back to CSV
    # lhd_df.to_csv(lhd_df_path, index=False)

    for gpkg_path in gpkg_set:  # goes through the unique set of GPKGs and downloads them
        gpkg_name = gpkg_path.split("/")[-1]
        vpu_id = gpkg_name[-8:-5]
        gpkg_url = f"{base_url}/vpu%3D{vpu_id}/{gpkg_name}"
        download_gpkg(gpkg_url, gpkg_path)

    return lhd_df

# Example usage
# lhd_df_path = "C:/Users/pgordi/Downloads/LHD_Download_Function_Test_Sheet_CSV.csv"
# gpkg_dir = "C:/Users/pgordi/Downloads"
# assign_flowlines(lhd_df_path, gpkg_dir)

