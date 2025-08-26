import os
import ast
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

"""
these functions extract data frames from raw files
"""


def run_successful(run_results_dir):
    id_no = os.path.basename(run_results_dir)
    xs_results = os.path.join(run_results_dir, "XS", f"{id_no}_Local_XS_Lines.gpkg")

    if not os.path.exists(xs_results):
        return False  # or raise an error if preferred

    try:
        xs_gdf = gpd.read_file(xs_results)
        return not xs_gdf.empty
    except Exception as e:
        print(f"Error reading DBF for {id_no}: {e}")
        return False


def fuzzy_merge(left, right, tol=2) -> pd.DataFrame:
    """
    Perform fuzzy merge based on Row and Col coordinates within tolerance
    """
    result_rows = []

    # Get column names to avoid conflicts
    right_cols_to_add = [col for col in right.columns if col not in ['COMID', 'Row', 'Col']]

    for comid, group_left in left.groupby('COMID'):
        group_right = right[right['COMID'] == comid].copy()

        if group_right.empty:
            # No matches for this COMID, add left rows with NaN for right columns
            for col in right_cols_to_add:
                group_left = group_left.copy()
                group_left[col] = np.nan
            result_rows.append(group_left)
            continue

        for idx, row_left in group_left.iterrows():
            # Find matches within tolerance
            row_diff = abs(group_right['Row'] - row_left['Row'])
            col_diff = abs(group_right['Col'] - row_left['Col'])
            matches = group_right[(row_diff <= tol) & (col_diff <= tol)]

            if not matches.empty:
                # If multiple matches, take the closest one
                if len(matches) > 1:
                    distances = row_diff + col_diff
                    closest_idx = distances.idxmin()
                    match = matches.loc[closest_idx]
                else:
                    match = matches.iloc[0]

                # Create combined row
                combined_row = row_left.copy()
                for col in right_cols_to_add:
                    combined_row[col] = match[col]
                result_rows.append(combined_row.to_frame().T)
            else:
                # No match found, add NaN values for right columns
                row_with_nans = row_left.copy()
                for col in right_cols_to_add:
                    row_with_nans[col] = np.nan
                result_rows.append(row_with_nans.to_frame().T)

    # Concatenate all results
    if result_rows:
        result_df = pd.concat(result_rows, ignore_index=True)
        return result_df
    else:
        return pd.DataFrame()


def merge_arc_results(curve_file: str, local_vdt: str, cross_section: str) -> gpd.GeoDataFrame|pd.DataFrame:
    # Read files
    vdt_gdf = gpd.read_file(local_vdt)
    rc_gdf = gpd.read_file(curve_file)
    xs_gdf = gpd.read_file(cross_section)

    # Convert list-like strings to actual lists with error handling
    list_columns = ['XS1_Profile', 'Manning_N_Raster1', 'XS2_Profile', 'Manning_N_Raster2']
    for col in list_columns:
        if col in xs_gdf.columns:
            xs_gdf[col] = xs_gdf[col].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else x)

    # Drop duplicate 'Ordinate_Dist.1' if it exists and is identical
    if 'Ordinate_Dist.1' in xs_gdf.columns and 'Ordinate_Dist' in xs_gdf.columns:
        if xs_gdf['Ordinate_Dist'].equals(xs_gdf['Ordinate_Dist.1']):
            xs_gdf = xs_gdf.drop(columns=['Ordinate_Dist.1'])

    # Perform fuzzy merge with XS data
    first_merge = fuzzy_merge(rc_gdf, vdt_gdf, tol=2)
    results_gdf = fuzzy_merge(first_merge, xs_gdf, tol=2)

    if 'geometry' in results_gdf.columns:
        results_gdf = gpd.GeoDataFrame(results_gdf, geometry='geometry')

    results_gdf = results_gdf.sort_values(by=["Row", "Col"]).reset_index(drop=True)

    if results_gdf['DEM_Elev'][0] < results_gdf['DEM_Elev'][len(results_gdf)-1]:
        # this means the cross-sections are going upstream, so let's switch 'em around
        results_gdf = results_gdf[::-1].reset_index(drop=True)

    return results_gdf


"""
these functions plot create plots from the data frames
"""

def plot_cross_sections(combined_gdf, output_dir):
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=len(combined_gdf)-1)

    for i in range(len(combined_gdf)):
        y_1 = combined_gdf['XS1_Profile'].iloc[i]  # since these go from the center out, i'll flip them around
        y_2 = combined_gdf['XS2_Profile'].iloc[i]
        x_1 = [0 - j * combined_gdf['Ordinate_Dist'].iloc[i] for j in range(len(y_1))]
        x_2 = [0 + j * combined_gdf['Ordinate_Dist'].iloc[i] for j in range(len(y_2))]

        INVALID_THRESHOLD = -1e5
        # delete any points that contain missing data
        x = x_1[::-1] + x_2
        y = y_1[::-1] + y_2

        # Filter out invalid values
        x_clean = []
        y_clean = []
        for xi, yi in zip(x, y):
            zip(x, y)
            if yi > INVALID_THRESHOLD:
                x_clean.append(xi)
                y_clean.append(yi)

        # create the plot
        color = cmap(norm(i))
        print(f'base elevation = {min(y_clean)}')

        if i > 0:
            plt.plot(x_clean, y_clean, label=f'Downstream Cross-section {i}',
                     color=color)
        else:
            plt.plot(x_clean, y_clean, label=f'Upstream Cross-section',
                     color=color)

    plt.legend(title="Cross-Sections", loc='best', fontsize='small')  # Add legend
    plt.ylabel('Elevation (m)')
    plt.xlabel('Lateral Distance (m)')
    png_output = os.path.join(output_dir, f'/Reach Cross-Sections at LHD No. {lhd_id}.png')
    plt.savefig(png_output, dpi=300, bbox_inches='tight')
    plt.show()


def plot_rating_curves(curve_file, output_dir):
    # define range of flows
    x = np.linspace(1, 1000, 100)

    # initialize the plot
    plt.figure(figsize=(10, 6))

    # iterate through the rating curves
    for index, row in curve_file.iterrows():
        a = row['depth_a']
        b = row['depth_b']
        y = a * x**b
        if index == 0:
            plt.plot(x, y, label=f'Upstream Rating Curve {index}: $y = {a:.3f} x^{{{b:.3f}}}$')
        else:
            plt.plot(x, y, label=f'Downstream Rating Curve No. {index}: $y = {a:.3f} x^{{{b:.3f}}}$')

    # Add labels and legend
    plt.xlabel('Flow (m$^{3}$/s)')
    plt.ylabel('Depth (m)')
    plt.title(f'Downstream Rating Curves at LHD No. {lhd_id}')
    plt.legend(title="Rating Curve Equations", loc='best', fontsize='small')  # Add legend
    plt.grid(True)

    # create output path for png file and save it
    png_output = os.path.join(output_dir, f'/Downstream Rating Curves at LHD No. {lhd_id}.png')
    plt.savefig(png_output, dpi=300, bbox_inches='tight')
    plt.show()


def plot_water_profiles(combined_gdf: gpd.GeoDataFrame, full_database_df: pd.DataFrame, output_dir: str=None, save: bool=False):
    plt.plot(full_database_df.index, full_database_df['DEM_Elev'], color='dodgerblue', label='DEM Elevation')
    upstream_xs = combined_gdf.iloc[0]
    upstream_row = upstream_xs['Row']
    upstream_col = upstream_xs['Col']

    # downstream_xs = combined_gdf.iloc[1]
    # downstream_row = downstream_xs['Row']
    # downstream_col = downstream_xs['Col']

    upstream_idx = full_database_df[(full_database_df["Row"] == upstream_row)
                               & (full_database_df["Col"] == upstream_col)].index[0]

    plt.scatter(upstream_idx, upstream_xs['DEM_Elev'], label=f'Upstream Elevation')

    # downstream_idx = database_df[(full_database_df["Row"] == downstream_row)
    #                              & (full_database_df["Col"] == downstream_col)].index

    for i in range(1, len(combined_gdf)):
        # let's get each downstream xs index, then we'll plot it's DEM_Elev to see if they're in the right order
        downstream_xs = combined_gdf.iloc[i]
        downstream_row = downstream_xs['Row']
        downstream_col = downstream_xs['Col']

        downstream_idx = full_database_df[(full_database_df["Row"] == downstream_row)
                                     & (full_database_df["Col"] == downstream_col)].index[0]

        plt.scatter(downstream_idx, downstream_xs['DEM_Elev'], label=f'Downstream Elevation No. {i}')


    # like danger zone... but with dam
    # damger_zone = database_df.loc[upstream_idx[0]:downstream_idx[0]]
    # damger_zone = database_df.loc[0:downstream_idx[0]]


    if save:
        png_output = os.path.join(output_dir, f'/Longitudinal Water Surface Profile at LHD No. {lhd_id}.png')
        plt.savefig(png_output, dpi=300, bbox_inches='tight')
    plt.legend()
    plt.title(label=f'DEM WSE at LHD No. {lhd_id}')
    plt.show()



def merge_databases(cf_database, xs_database):
    cf_df = pd.read_csv(cf_database)
    xs_df = pd.read_csv(xs_database, sep='\t')

    return pd.merge(xs_df, cf_df, on=['COMID', 'Row', 'Col'])



"""
this is the real deal... at least a real test case
"""


# this folder has the results from some example runs
all_results = "E:/LHD_1-m_NWM/LHD_Results"
# these are the subdirectories for each rathcelon run
rath_runs = [os.path.join(all_results, d) for d in os.listdir(all_results) if os.path.isdir(os.path.join(all_results, d))]

for rath_run in rath_runs:
    # the lhd_id is the name of each directory
    if run_successful(rath_run):
        lhd_id = os.path.basename(rath_run)
        # read in the cf and xs file...
        cf_gpkg = os.path.join(rath_run, 'VDT', f'{lhd_id}_Local_CurveFile.gpkg')
        xs_gpkg = os.path.join(rath_run, 'XS', f'{lhd_id}_Local_XS_Lines.gpkg')
        vdt_gpkg = os.path.join(rath_run, 'VDT', f'{lhd_id}_Local_VDT_Database.gpkg')

        cf_csv = os.path.join(rath_run, 'VDT', f'{lhd_id}_CurveFile.csv')
        xs_txt = os.path.join(rath_run, 'XS', f'{lhd_id}_XS_Out.txt')

        merged_gdf = merge_arc_results(cf_gpkg, vdt_gpkg, xs_gpkg)
        database_df = merge_databases(cf_csv, xs_txt)

        # plot rating curves and cross-sections
        # plot_rating_curves(merged_gdf, rath_run)
        # plot_cross_sections(merged_gdf, rath_run)
        plot_water_profiles(merged_gdf, database_df)
