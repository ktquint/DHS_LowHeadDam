import requests
import pandas as pd


def download_gpkg(gpkg_url, local_path):
    """
    downloads the geoglows gpkgs locally
    """
    response = requests.get(gpkg_url)
    with open(local_path, 'wb') as f:
        f.write(response.content)


def find_col_name(df, value):
    for col in df.columns:
        if value in df[col].values:
            return col


def assign_flowlines(lhd_df, gpkg_dir):
    """
    this
    lhd_df: dataframe with lat, long, dem, etc. info
    gpkg_dir: directory where gpkg will be downloaded
    """
    lhd_df["flowline"] = ""
    base_url = "https://geoglows-v2.s3-us-west-2.amazonaws.com/streams"
    gpkg_set = set()
    linkno_df = pd.read_csv("C:/Users/ki87ujmn/PycharmProjects/DHS_LowHeadDam/list_of_linkno.csv")

    for index, row in lhd_df.iterrows():
        linkno = row.LINKNO # get the linkno for each dam
        gpkg_name = find_col_name(linkno_df, linkno) # find the gpkg with the linkno
        gpkg_path = f"{gpkg_dir}/{gpkg_name}.gpkg"
        lhd_df.at[index, 'flowline'] = gpkg_path # save the gpkg path for each dam
        gpkg_set.add(gpkg_path)

    for gpkg_path in gpkg_set: # goes through the unique set of gpkgs and downloads them
        """
        esta no funciona :(
        """

        gpkg_name = gpkg_path.split("/")[-1]
        gpkg_url = f"{base_url}/{gpkg_name}"
        download_gpkg(gpkg_url, gpkg_path)
    return lhd_df
