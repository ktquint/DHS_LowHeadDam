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


def run_successful(run_results_dir):
    lhd_id = os.path.basename(run_results_dir)
    dbf_path = os.path.join(run_results_dir, "VDT", f"{lhd_id}_Local_VDT_Database.dbf")

    if not os.path.exists(dbf_path):
        return False  # or raise an error if preferred

    try:
        local_vdt_dbf = DBF(dbf_path)
        local_vdt_df = pd.DataFrame(iter(local_vdt_dbf))
        return not local_vdt_df.empty
    except Exception as e:
        print(f"Error reading DBF for {lhd_id}: {e}")
        return False


def merge_vdt_xs(results_dir, lhd_id):
    # local vdt database
    vdt_local = DBF(os.path.join(results_dir, "VDT", f"{lhd_id}_Local_VDT_Database.dbf"))
    # local curve file
    cf_local = DBF(os.path.join(results_dir, "VDT", f"{lhd_id}_Local_CurveFile.dbf"))
    # vdt database
    vdt_database = os.path.join(results_dir, "VDT", f"{lhd_id}_VDT_Database.txt")
    # cross-section databaseLiteralString | str | bytes
    xs_database = os.path.join(results_dir, "XS", f"{lhd_id}_XS_Out.txt")

    # now we'll turn these files into dataframes
    local_vdt_df = pd.DataFrame(iter(vdt_local))
    cf_df = pd.DataFrame(iter(cf_local))
    xs_df = pd.read_csv(xs_database, header=None, sep='\t')
    xs_df.rename(columns={0: 'COMID', 1: 'Row', 2: 'Col', 3: 'elev_a',
                          4: 'wse_a', 5: 'lat_a', 6: 'n_a', 7: 'elev_b',
                          8: 'wse_b', 9: 'lat_b', 10: 'n_b', 11: 'slope'},
                 inplace=True)
    # evaluate the strings as literals (lists)
    xs_df['elev_a'] = xs_df['elev_a'].apply(ast.literal_eval)
    xs_df['n_a'] = xs_df['n_a'].apply(ast.literal_eval)
    xs_df['elev_b'] = xs_df['elev_b'].apply(ast.literal_eval)
    xs_df['n_b'] = xs_df['n_b'].apply(ast.literal_eval)
    # the vdt database is a csv, but saved as a txt file
    vdt_df = pd.read_csv(vdt_database)

    # our first merge will give us a giant database with all the vdt data and cross-section data
    merge_1 = pd.merge(vdt_df, xs_df, on=['COMID', 'Row', 'Col'], how='left')
    # out second merge will cut it down to just the rows and columns in the curve file
    merge_2 = pd.merge(cf_df, merge_1, on=['COMID', 'Row', 'Col'], how='left')

    return merge_2


def create_xs_figs(xs_1m, xs_10m, lhd_id):
    for i in range(0, len(xs_1m)):
        # first let's extract the data from the 1-m data
        y_1 = xs_1m.at[i, 'elev_a']
        y_1 = y_1[::-1]
        y_2 = xs_1m.at[i, 'elev_b']
        xs_elevation_1 = y_1 + y_2

        x_1 = [0 + j * xs_1m.at[i, 'lat_a'] for j in range(len(y_1))]
        x_2 = [max(x_1) + j * xs_1m.at[i, 'lat_b'] for j in range(len(y_2))]
        xs_lateral_1 = x_1 + x_2
        plt.plot(xs_lateral_1, xs_elevation_1, label=f'1-m Resolution Cross-section')
        # next we'll get the info from the 10-m data
        y_1 = xs_10m.at[i, 'elev_a']
        y_1 = y_1[::-1]
        y_2 = xs_10m.at[i, 'elev_b']
        xs_elevation_10 = y_1 + y_2

        x_1 = [0 + j * xs_10m.at[i, 'lat_a'] for j in range(len(y_1))]
        x_2 = [max(x_1) + j * xs_10m.at[i, 'lat_b'] for j in range(len(y_2))]
        xs_lateral_10 = x_1 + x_2
        plt.plot(xs_lateral_10, xs_elevation_10, label=f'10-m Resolution Cross-section')

        plt.xlabel('Lateral Distance (m)')
        plt.ylabel('Elevation (m)')
        plt.title(f'Cross-Sections for LHD No. {lhd_id}')
        plt.legend()
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()


def create_rating_curves(rc_1m, rc_10m, lhd_id):

    for i in range(0, len(rc_1m)):
        q_1m = []
        d_1m = []
        q_10m = []
        d_10m = []
        for j in range(1, 31):
            if rc_1m.at[i, f"wse_{j}"] > 0:
                q_1m.append(rc_1m.at[i, f"q_{j}"])
                d_1m.append(rc_1m.at[i, f"wse_{j}"] - rc_1m.at[i, 'elev_a'][0])

                q_10m.append(rc_10m.at[i, f"q_{j}"])
                d_10m.append(rc_10m.at[i, f"wse_{j}"] - rc_10m.at[i, 'elev_a'][0])

        Q = np.linspace(1, q_1m[-1], 100)
        y_1m = rc_1m.at[i, 'depth_a'] * Q ** rc_1m.at[i, 'depth_b']
        plt.plot(Q, y_1m, label="CF 1-m", color='orange')
        plt.plot(q_1m, d_1m, 'o', label="VDT 1-m", color='red')

        Q = np.linspace(1, q_10m[-1], 100)
        y_10m = rc_10m.at[i, 'depth_a'] * Q ** rc_10m.at[i, 'depth_b']
        plt.plot(Q, y_10m, label="CF 1-m", color='green')
        plt.plot(q_10m, d_10m, 'o', label="VDT 1-m", color='dodgerblue')

        plt.xlabel('Flow (m$^{3}$/s)')
        plt.ylabel('Depth (m)')
        plt.grid(True)
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.legend()
        plt.title(label=f'Rating Curves for LHD No. {lhd_id}')
        plt.show(block=False)
        plt.pause(0.1)
        plt.close()


def compare_1m_to_10m(one_meter_results, ten_meter_results):
    """
        inputs are file paths (e.g., LHD_Project/LHD_Results and LHD_1m/LHD_Results)
    """
    # first lets list out the dams we have results for in each set
    one_meter_runs = [d for d in os.listdir(one_meter_results)]
    ten_meter_runs = [d for d in os.listdir(ten_meter_results)]
    # then we'll trim down to just the runs we have in common
    common_runs = list(set(one_meter_runs).intersection(set(ten_meter_runs)))
    print(common_runs)

    # okay, let's loop through each run and do some magic...
    for lhd_id in common_runs:
        one_meter_dir = os.path.join(one_meter_results, lhd_id)
        ten_meter_dir = os.path.join(ten_meter_results, lhd_id)
        if run_successful(one_meter_dir) and run_successful(ten_meter_dir):
            print(f"Working on {lhd_id}")
            one_meter_data = merge_vdt_xs(one_meter_dir, lhd_id)
            ten_meter_data = merge_vdt_xs(ten_meter_dir, lhd_id)
            create_xs_figs(one_meter_data, ten_meter_data, lhd_id)
            create_rating_curves(one_meter_data, ten_meter_data, lhd_id)


def count_rows_in_dbf(file_path):
    try:
        table = DBF(file_path)
        row_count = len(table)
        print(f"Number of rows in '{file_path}': {row_count}")
        return row_count
    except Exception as e:
        print(f"Error reading DBF file: {e}")


def count_good_files(results_dir):
    rath_runs = [d for d in os.listdir(results_dir)]

    # okay, let's loop through each run and do some magic...
    for lhd_id in rath_runs:
        lhd_result = os.path.join(results_dir, lhd_id)
        dbf_path = os.path.join(lhd_result, "VDT", f"{lhd_id}_Local_CurveFile.dbf")
        if run_successful(lhd_result):
            print(f"{lhd_id} was ran successfully")
            print(count_rows_in_dbf(dbf_path))


# compare_1m_to_10m("E:\LHD_Project\LHD_Results", "E:\LHD_1-3_arc-second - Copy\LHD_Results")
count_good_files("E:\LHD_1-3_arc-second - Copy\LHD_Results")


