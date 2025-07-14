import os
import s3fs
import requests
import pandas as pd


def download_NHDPlus(local_path):
    """
        i'll fill this out later...
    """
    return local_path


def download_geoglows(gpkg_url, local_path):
    """
        Downloads the GeoGLows GPKGs locally.
    """
    if not os.path.exists(local_path):
        try:
            fs = s3fs.S3FileSystem(anon=True)
            # Download a file
            with fs.open(gpkg_url, 'rb') as f_in:
                with open(local_path, 'wb') as f_out:
                    f_out.write(f_in.read())
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {gpkg_url}: {e}")
    else:
        print(f"Local file {local_path} already exists")


def find_col_name(df, value):
    for col in df.columns:
        if value in df[col].values:
            return col


def assign_flowlines(lhd_df, gpkg_dir, data_source):
    """
        Assigns flowlines to the LHD dataframe and downloads the necessary GPKGs.

        lhd_df_path: Path to the CSV file with lat, long, dem, etc. info.
        gpkg_dir: Directory where GPKGs will be downloaded.
    """
    if data_source == "GEOGLOWS":
        lhd_df["flowline"] = None
        base_url = 'geoglows-v2/hydrography/'
        gpkg_set = set()

        metadata_df = pd.read_parquet("C:/Users/ki87ujmn/PycharmProjects/DHS_LowHeadDam/geoglows_metadata.parquet")
        metadata_df['LINKNO'] = metadata_df['LINKNO'].astype(int)

        print("Starting Hydrography Download Process...")

        for index, row in lhd_df.iterrows():
            linkno = row['LINKNO'] # get the linkno for each dam

            # Lookup VPUCode for this linkno
            row = metadata_df[metadata_df['LINKNO'] == linkno]
            if row.empty:
                raise ValueError(f"linkno {linkno} not found in metadata.")

            vpu_code = str(row.iloc[0]['VPUCode']).zfill(2)  # Ensure leading 0s

            gpkg_path = os.path.join(gpkg_dir, f"streams_{vpu_code}.gpkg")
            gpkg_set.add(gpkg_path)

        # Save updated DataFrame back to CSV
        # lhd_df.to_csv(lhd_df_path, index=False)

        for gpkg_path in gpkg_set:  # goes through the unique set of GPKGs and downloads them
            gpkg_name = gpkg_path.split("/")[-1]
            vpu_id = gpkg_name[-8:-5]
            gpkg_url = f"{base_url}vpu={vpu_id}/{gpkg_name}"
            download_geoglows(gpkg_url, gpkg_path)
    else:
        print("Data source not specified")
    return lhd_df
