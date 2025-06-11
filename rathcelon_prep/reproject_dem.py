import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling


def reproject_dem_to_nad83(src_path, dst_crs="EPSG:4269"):
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        # Use a temporary file for safety
        temp_path = src_path.replace(".tif", "_tmp.tif")

        with rasterio.open(temp_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear)

    # Overwrite the original file
    os.replace(temp_path, src_path)


def batch_reproject_dems(parent_folder):
    for subdir, dirs, files in os.walk(parent_folder):
        for file in files:
            if file.lower().endswith(".tif"):
                input_path = os.path.join(subdir, file)
                print(f"Reprojecting: {input_path}")
                reproject_dem_to_nad83(input_path)


# Example usage
batch_reproject_dems(r"E:\LHD_Project\LHD_DEMs")


import os
import ast
import numpy as np
import pandas as pd
# noinspection PyProtectedMember
from dbfread import DBF
import matplotlib.pyplot as plt


def plot_water_surface_elevations(df_row):
    lhd_id = df_row['ID']
    y_1 = df_row['elev_a']
    y_1 = y_1[::-1]
    y_2 = df_row['elev_b']
    xs_elevation = y_1 + y_2

    x_1 = [0 + j * df_row['lat_a'] for j in range(len(y_1))]
    x_2 = [max(x_1) + j * df_row['lat_b'] for j in range(len(y_2))]
    xs_lateral = x_1 + x_2
    for i in range(1, 31):
        wse_i = df_row[f'wse_{i}']
        wse_list = [wse_i, wse_i]
        lateral_list = [min(xs_lateral), max(xs_lateral)]
        plt.plot(lateral_list, wse_list, color='dodgerblue', linestyle='--')
    plt.plot(xs_lateral, xs_elevation, color='black')
    plt.xlabel('Lateral Distance (m)')
    plt.ylabel('Elevation (m)')
    plt.title(f'Water Surface Elevation at LHD No. {lhd_id}')
    plt.show()


def plot_downstream(cf_local, vdt_database, xs_database, id):
    """
        all inputs are dataframes
    """
    row_col = cf_local['Row'].tolist()
    col_col = cf_local['Col'].tolist()

    indices = []
    for i in range(len(row_col)):
        row, col = row_col[i], col_col[i]
        index = vdt_database[(vdt_database['Row'] == row) & (vdt_database['Col'] == col)].index[0]
        indices.append(index)

    start_index = min(indices)
    end_index = max(indices)




    vdt_trimmed = vdt_database.iloc[start_index:end_index + 1]  # +1 to include the end row
    xs_trimmed = xs_database.iloc[start_index:end_index + 1]

    downstream_dist = [0]
    total_dist = 0
    wse = vdt_trimmed['Elev'].tolist()
    row_list = vdt_database['Row'].tolist()
    col_list = vdt_database['Col'].tolist()
    xs_1 = xs_trimmed['elev_a'].tolist()
    xs_2 = xs_trimmed['elev_b'].tolist()
    bed_elev = [min(xs_1[0] + xs_2[0])]

    for i in range(len(vdt_trimmed['Row'].tolist()) - 1):
        min_elev = min(xs_1[i + 1] + xs_2[i + 1])
        bed_elev.append(min_elev)
        dist = (row_list[i + 1] - row_list[i]) ** 2 + (col_list[i + 1] - col_list[i]) ** 2
        total_dist += np.sqrt(dist)
        downstream_dist.append(total_dist)
    wse = wse[::-1]
    bed_elev = bed_elev[::-1]

    plt.plot(downstream_dist, wse, color='dodgerblue', label='Water Surface')
    plt.plot(downstream_dist, bed_elev, color='black', label='Estimated Bed Elevation')
    plt.xlabel('Downstream Distance (m)')
    plt.ylabel('Elevation (m)')
    plt.legend()
    plt.title(f"Water Surface Elevation Along Streamline at LHD No. {id}")
    plt.show()



def plot_wse_bed_elevations(lhd_csv):
    project_dir = os.path.dirname(lhd_csv)
    lhd_df = pd.read_csv(lhd_csv)

    for index, row in lhd_df.iterrows():
        lhd_id = str(row['ID'])
        XS_dir = os.path.join(project_dir, "LHD_Results", lhd_id, "XS")
        VDT_dir = os.path.join(project_dir, "LHD_Results", lhd_id, "VDT")

        XS_txt = os.path.join(XS_dir, f"{lhd_id}_XS_Out.txt")
        local_VDT_dbf = DBF(os.path.join(VDT_dir, f"{lhd_id}_Local_VDT_Database.dbf"))
        local_CF_dbf = DBF(os.path.join(VDT_dir, f"{lhd_id}_Local_CurveFile.dbf"))
        VDT_txt = os.path.join(VDT_dir, f"{lhd_id}_VDT_Database.txt")

        # local_vdf_df = pd.DataFrame(iter(local_VDT_dbf))
        local_cf_df = pd.DataFrame(iter(local_CF_dbf))

        xs_df = pd.read_csv(XS_txt, header=None, sep='\t')
        xs_df.rename(columns={0: 'COMID', 1: 'Row', 2: 'Col', 3: 'elev_a',
                              4: 'wse_a', 5: 'lat_a', 6: 'n_a', 7: 'elev_b',
                              8: 'wse_b', 9: 'lat_b', 10: 'n_b', 11: 'slope'},
                     inplace=True)
        # evaluate the strings as literals (lists)
        xs_df['elev_a'] = xs_df['elev_a'].apply(ast.literal_eval)
        xs_df['n_a'] = xs_df['n_a'].apply(ast.literal_eval)
        xs_df['elev_b'] = xs_df['elev_b'].apply(ast.literal_eval)
        xs_df['n_b'] = xs_df['n_b'].apply(ast.literal_eval)
        vdt_df = pd.read_csv(VDT_txt)

        merged_df = pd.merge(vdt_df, xs_df, on=['COMID', 'Row', 'Col'], how='left')
        # local_cf_df.drop(['COMID', 'Row', 'Col', 'Lat', 'Lon'], axis=1, inplace=True)
        # final_df = pd.merge(local_cf_df, merged_df, on=['COMID', 'Row', 'Col'], how='left')
        # final_df = pd.concat([merged_df.reset_index(drop=True),
        #                       local_cf_df.reset_index(drop=True)], axis=1)
        plot_downstream(local_cf_df, vdt_df, xs_df, lhd_id)
        # for index, row in final_df.iterrows():
        #     print(f'Plotting the {index}th set')
        #     plot_water_surface_elevations(row)



plot_wse_bed_elevations("E:\LHD_1-3_arc-second\LowHead_Dam_Database.csv")