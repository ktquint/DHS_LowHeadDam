"""
This file holds all the objects we'll be using

CrossSections holds all the information we have associated with a given cross-section. The objects will be made using
the ID_XS_Out.txt file for each dam

Dam holds the cross-section information
"""

import os               # interacts with the operating system (e.g., file paths, directories)
import ast              # safely parses Python code or literals (e.g., safe string-to-list conversion)
import math             # Standard math functions (trigonometry, logarithms, rounding, etc.)
import pyproj           # handles CRS conversions
import rasterio         # reads, writes, and processes raster datasets
import numpy as np      # numerical computing with support for arrays, matrices, and math functions
import pandas as pd     # data analysis and manipulation tool; reads in CSV, TXT, XLSX files
import dbfread as dbf   # reads DBF (dBase) files (the rathcelon output)
import geopandas as gpd # extends pandas for working with geospatial vector data (points, lines, polygons)
import contextily as ctx    # adds basemaps (e.g., OpenStreetMap) to geospatial plots, often used with geopandas
import hydraulics as hyd    # hydraulic calculations and functions (my own code)
import hydroinformatics as hi   # access streamflow information (retrospective and return period flows)
from rasterio.plot import show  # displays raster datasets with matplotlib
import matplotlib.pyplot as plt # creates static, animated, and interactive plots and graphs
from rasterio.io import MemoryFile  # allows in-memory reading/writing of raster files without saving to disk
from matplotlib.ticker import FixedLocator  # custom axis tick locations in plots
from rasterio.warp import calculate_default_transform, reproject, Resampling
                                            # functions for reprojection and resampling of raster datasets
                                            # (coordinate transforms, resolution changes)


def round_sigfig(num, sig_figs):
    if num == 0:
        return 0
    else:
        return round(num, sig_figs - int(math.floor(math.log10(abs(num)))) - 1)


class CrossSection:
    def __init__(self, index, xs_row, dam_id, weir_length, fig_dir):
        """
            lateral is a str of lateral distances
            elevation is a str of elevations
            wse is water surface elevation as a float
            distance is the downstream or upstream distance from the dam
            location is either 'Downstream' or 'Upstream'

            rating curve eq.: d = a * Q **b
                        where 'depth_a' is a & 'depth_b' is b
        """
        self.id = dam_id
        self.fig_dir = fig_dir # all the figs we make for each cross-section will be saved here

        # geospatial info
        self.lat = xs_row['Lat']
        self.lon = xs_row['Lon']
        self.L = weir_length
        self.index = index
        if self.index == 4:
            self.location = 'Upstream'
            # upstream cross-section is 1/2 the weir length from the dam
            self.distance = int(self.L / 2)
        else:
            self.location = 'Downstream'
            # downstream cross-sections are one weir length farther than the previous
            self.distance = int((self.index + 1) * self.L)

        # rating curve info
        self.a = xs_row['depth_a']
        self.b = xs_row['depth_b']
        self.max_Q = xs_row['QMax']
        self.slope = round_sigfig(xs_row['Slope'], 3)
        self.fatal_qs = None # np.array(ast.literal_eval(id_row['fatality_flows'].values[0]))

        # cross-section plot info
        self.wse = xs_row['Elev']
        # i'll make two lists that i'll populate with just the wse values... you'll see why in a minute
        INVALID_THRESHOLD = -1e5
        y_1 = xs_row['XS1_Profile'][::-1] # since these go from the center out, i'll flip them around
        y_2 = xs_row['XS2_Profile']
        x_1 = [0 + j * xs_row['Ordinate_Dist'] for j in range(len(y_1))]
        x_2 = [max(x_1) + j * xs_row['Ordinate_Dist'] for j in range(len(y_2))]

        # delete any points that contain missing data
        x = x_1 + x_2
        y = y_1 + y_2

        # Filter out invalid values
        x_clean = []
        y_clean = []
        for xi, yi in zip(x, y):
            if yi > INVALID_THRESHOLD:
                x_clean.append(xi)
                y_clean.append(yi)

        self.elevation = y_clean
        self.lateral = x_clean
        # let's start with the left side
        wse_left = []
        wse_lat_left = []
        # we'll use a while loop
        # on the left side we'll start with the right-most element
        i = len(y_1) - 1
        while y_1[i] <= self.wse:
            wse_left.append(self.wse)
            wse_lat_left.append(x_1[i])
            i -= 1

        # now the right side
        wse_right = []
        wse_lat_right = []
        i = 0
        while y_2[i] <= self.wse:
            wse_right.append(self.wse)
            wse_lat_right.append(x_2[i])
            i += 1

        self.water_elevation = wse_left + wse_right
        self.water_lateral = wse_lat_left + wse_lat_right

        # initialize P, but don't give it a value yet
        self.P = None


    def set_dam_height(self, P):
        """
        pretty self-explanatory, no?
        """
        self.P = P


    def set_fatal_qs(self, q_list):
        """
        same explanation as set_dam_height
        """
        self.fatal_qs = np.array(q_list)


    def plot_cross_section(self):
        """
            creates a figure for each cross-section and saves it in the figs folder within th results folder
        """
        fig, ax = plt.subplots()
        # cross-section elevations
        ax.plot(self.lateral, self.elevation,
                 color='black', label=f'Downstream Slope: {self.slope}')

        # wse line
        ax.plot(self.water_lateral, self.water_elevation,
                 color='cyan', linestyle='--', label=f'Water Surface Elevation: {self.wse} m')
        ax.set_xlabel('Lateral Distance (m)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title(f'{self.location} Cross-Section {self.distance} meters from LHD No. {self.id}')
        ax.legend(loc='upper right')
        # the file name stands for Downstream/Upstream Cross-section No. XX at Low-Head Dam No. XX
        if self.index == 4:
            location = 'US'
            fig_loc = os.path.join(self.fig_dir, f"{location}_XS_LHD_{self.id}.png")
        else:
            location = 'DS'
            fig_loc = os.path.join(self.fig_dir, f"{location}_XS_{self.index+1}_LHD_{self.id}.png")
        fig.savefig(fig_loc)
        return fig


    def create_rating_curve(self):
        x = np.linspace(0.01, self.max_Q, 100)
        y = self.a * x ** self.b
        plt.plot(x, y,
                 label=f'Rating Curve {self.distance} meters {self.location}: $y = {self.a:.3f} x^{{{self.b:.3f}}}$')


    def plot_rating_curve(self):
        x = np.linspace(0.01, self.max_Q, 100)
        y = self.a * x ** self.b
        plt.plot(x, y, color='black', label=f'$y = {self.a:.3f} x^{{{self.b:.3f}}}$')

        plt.xlabel('Flow (m$^{3}$/s)')
        plt.ylabel('Depth (m)')
        plt.title(f'{self.location} Rating Curve {self.distance} meters from LHD No. {self.id}')
        plt.legend(title=f'{self.location} Rating Curve Equation')

        plt.grid(True)
        plt.show()


    def plot_flip_sequent(self, ax):
        # set the range of Q's we want to plot
        Qs = np.linspace(0.01, self.max_Q, 100)

        Y_Ts = self.a * Qs ** self.b
        Y_Flips = []
        Y_Conjugates = []

        for Q in Qs:
            Y_Flip, Y_Conj = hyd.compute_flip_and_conjugate(Q, self.L, self.P)
            Y_Flips.append(Y_Flip)
            Y_Conjugates.append(Y_Conj)

        Y_Flips = np.array(Y_Flips)
        Y_Conjugates = np.array(Y_Conjugates)

        # american units *eagle screech*... if you don't like it then just get rid of the unit conversions
        ax.plot(Qs*35.315, Y_Flips*3.281,
                 label="Flip Depth", color='gray', linestyle='--')
        ax.plot(Qs*35.315, Y_Ts*3.281,
                 label="Tailwater Depth", color='dodgerblue', linestyle='-')
        ax.plot(Qs*35.315, Y_Conjugates*3.281,
                 label="Sequent Depth", color='gray', linestyle='-')

        # make the plot look more presentable
        ax.grid(True)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        # add labels and title then show
        ax.set_xlabel('Discharge (ft$^{3}$/s)')
        ax.set_ylabel('Depth (ft)')
        # plt.xlabel('Discharge (m$^{3}$/s)')
        # plt.ylabel('Depth (m)')

        ax.set_title(f'Submerged Hydraulic Jumps at Low-Head Dam No. {self.id}')
        ax.legend(loc='upper left')

    def plot_fatal_flows(self, ax):
        fatal_m = self.a * self.fatal_qs**self.b
        ax.scatter(self.fatal_qs * 35.315, fatal_m * 3.281,
                 label="Recorded Fatality", marker='o',
                 facecolors='none', edgecolors='black')

    # noinspection PyTypeChecker
    def create_combined_fig(self):
        fig, ax = plt.subplots()
        self.plot_flip_sequent(ax)
        self.plot_fatal_flows(ax)
        # the file name stands for Rating Curve No. XX at Low-Head Dam No. XX
        fig_loc = os.path.join(self.fig_dir, f"RC_{self.index+1}_LHD_{self.id}.png")
        fig.savefig(fig_loc)
        return fig


class Dam:
    """
    create a Dam based on the BYU LHD IDs
    add cross-sections with information from vdt & cross_section files
    """
    def __init__(self, lhd_id, lhd_csv, project_dir, hydrology, est_dam):
        #database information
        self.id = int(lhd_id)
        lhd_df = pd.read_csv(lhd_csv)

        # create a folder to store figures...
        results_dir = os.path.join(project_dir, "LHD_Results", str(self.id)) # we'll use this a lot later
        self.fig_dir = os.path.join(results_dir, "FIGS")
        os.makedirs(self.fig_dir, exist_ok=True)
        id_row = lhd_df[lhd_df['ID'] == self.id].reset_index(drop=True)

        if est_dam:
            # if we need to estimate dam height, we'll create guesses for each cross-section
            P_guesses = ['P_1', 'P_2', 'P_3', 'P_4']
            slope_est = ['s_1', 's_2', 's_3', 's_4']
            for column in P_guesses:
                if column not in lhd_df.columns:
                    lhd_df[column] = None
            for column in slope_est:
                if column not in lhd_df.columns:
                    lhd_df[column] = None
            self.P = None
        else:
            self.P = id_row['P_known'].values[0] / 3.218

        # geographic information
        self.latitude = id_row['latitude'].values[0]
        self.longitude = id_row['longitude'].values[0]

        # physical information
        self.cross_sections = []
        self.weir_length = id_row['weir_length'].values[0]


        # fatality dates and fatal flows

        date_string = id_row['fatality_dates'].iloc[0]
        dates = date_string.strip("[]").split(", ")

        formatted_dates = []

        for date in dates:
            try:
                # Try converting full date (MM/DD/YYYY → YYYY-MM-DD)
                formatted_date = pd.to_datetime(date.strip(), format="%m/%d/%Y").strftime("%Y-%m-%d")
            except ValueError:
                try:
                    # Try converting partial date (MM/YYYY → YYYY-MM)
                    _ = pd.to_datetime(date.strip(), format="%m/%Y").strftime("%Y-%m")
                    continue
                except ValueError:
                    # If neither format applies, keep original
                    continue

            formatted_dates.append(formatted_date)

        self.fatality_dates = formatted_dates
        print(self.fatality_dates)

        # ----------------------------------------- HYDROLOGIC INFORMATION ------------------------------------------- #

        fatal_flows = []
        if hydrology == "GEOGLOWS":
            # establish which values to use as the comid
            self.comid = id_row['LINKNO'].values[0]

        elif hydrology == "USGS":
            fatal_flows = id_row['USGS_fatal_flows'].apply(ast.literal_eval).values[0]
            self.fatal_flows = [Q / 35.315 for Q in fatal_flows]

        else:   # NWM
            self.comid = id_row['reach_id'].values[0]

        # check to see if we already have a fatal flows column
        if 'fatal_flows' not in lhd_df.columns:
            lhd_df['fatal_flows'] = None

        # if the fatal_flows column is empty in the row we'll populate it
        elif pd.isna(id_row['fatal_flows'].values[0]):

            for date in self.fatality_dates:
                flow_value = hi.get_streamflow(hydrology, self.comid, [date, date])

                try:
                    # convert the np.float to a float
                    float_flow = float(flow_value)
                    fatal_flows.append(float_flow)

                except (ValueError, TypeError):
                    continue

            self.fatal_flows = fatal_flows
            # if we're getting them for the first time we'll save them to the csv
            lhd_df.loc[lhd_df['ID'] == self.id, 'fatal_flows'] = str(self.fatal_flows)

        # if there are already values we'll add them to our local variable
        else:
            self.fatal_flows = ast.literal_eval(id_row.at[0, 'fatal_flows'])

        # find attributes based on the vdt and xs files
        local_loc = os.path.join(results_dir, "VDT", f"{str(self.id)}_Local_CurveFile.dbf")
        database_loc = os.path.join(results_dir, "VDT", f"{str(self.id)}_Local_VDT_Database.dbf")
        xs_loc = os.path.join(results_dir, "XS", f"{str(self.id)}_XS_Out.txt" )

        # save tif and shp files for later...
        self.shp_loc = os.path.join(results_dir, "VDT", f"{str(self.id)}_Local_CurveFile.shp")
        self.tif_loc = os.path.join(results_dir, "Bathymetry", f"{str(self.id)}_ARC_Bathy.tif")

        # read in dbf then translate it to data.frame
        local_dbf = dbf.DBF(local_loc)
        local_df = pd.DataFrame(iter(local_dbf))
        database_dbf = dbf.DBF(database_loc)
        database_df = pd.DataFrame(iter(database_dbf))

        # read in txt as data.frame
        xs_df = pd.read_csv(xs_loc, sep='\t')

        # evaluate the strings as literals (lists)
        list_columns = ['XS1_Profile', 'Manning_N_Raster1', 'XS2_Profile', 'Manning_N_Raster2']
        for col in list_columns:
            xs_df[col] = xs_df[col].apply(ast.literal_eval)
        # check if the two 'Ordinate_Dist' columns are equal row-by-row
        if xs_df['Ordinate_Dist'].equals(xs_df['Ordinate_Dist.1']):
            xs_df = xs_df.drop(columns=['Ordinate_Dist.1'])  # drop the duplicate

        # let's merge the tables (how='left' because vdt only contains the xs we want and xs has all of them)
        merged_df = pd.merge(local_df, xs_df, on=['COMID', 'Row', 'Col'], how='left')

        database_df.drop(['COMID', 'Row', 'Col', 'Lat', 'Lon'], axis=1, inplace=True)

        complete_df = pd.concat([merged_df, database_df], axis=1)

        # let's go through each row of the df and create cross-sections objects
        for index, row in complete_df.iterrows():
            self.cross_sections.append(CrossSection(index, row, self.id, self.weir_length, self.fig_dir))

        # hydrologic information
        self.known_baseflow = id_row['known_baseflow'].values[0]

        # let's add the dam height and slope to the csv
        for i in range(len(self.cross_sections)-1):
            # add fatal qs to each cross-section
            self.cross_sections[i].set_fatal_qs(self.fatal_flows)
            # add slope info to csv file--it's not too important
            s_i = self.cross_sections[i].slope
            lhd_df.loc[lhd_df['ID'] == self.id, f's_{i + 1}'] = s_i
            y_ts = []
            y_flips = []
            y_2s = []

            if est_dam:
                delta_wse_i = complete_df.iloc[-1]['Elev'] - complete_df.at[i, 'Elev']
                # delta_wse_i = self.cross_sections[-1].wse - self.cross_sections[i].wse

                # tailwater using the power function
                # y_i = self.cross_sections[i].a * Q_i**self.cross_sections[i].b
                # tailwater using the wse and depth
                y_i = complete_df.at[i, 'Elev'] - min(self.cross_sections[i].elevation)
                # y_i = self.cross_sections[i].wse - min(self.cross_sections[i].elevation)

                # estimate dam height, add to cross-section and csv file
                z_i = -1 * self.cross_sections[i].slope * self.cross_sections[i].distance
                # P_i = dam_height(Q_i, L_i, delta_wse_i, y_i, z_i)
                # ths one has delta z = 0
                P_i = hyd.dam_height(self.known_baseflow, self.weir_length, delta_wse_i, y_i, z_i)
                if P_i < 1 or P_i > 100:
                    P_i = 1 # 3.05 # according to the literature, this is one of the most common dam heights
                self.cross_sections[i].set_dam_height(P_i)
                lhd_df.loc[lhd_df['ID'] == self.id, f'P_{i + 1}'] = P_i * 3.281 # convert to ft

                # now let's add the tailwater, flip, and conjugate depths for each flow/cross-section combo

                for flow in self.fatal_flows:
                    # calc. tailwater from power function
                    y_t = self.cross_sections[i].a * flow**self.cross_sections[i].b

                    # calc conj. and flip using leuthusser eq.s
                    y_flip, y_2 = hyd.compute_flip_and_conjugate(flow, self.weir_length, P_i)

                    # add those depths to their respective lists
                    y_ts.append(float(y_t))
                    y_flips.append(float(y_flip))
                    y_2s.append(float(y_2))
            else:
                self.cross_sections[i].set_dam_height(self.P)
                for flow in self.fatal_flows:
                    y_t = self.cross_sections[i].a * flow ** self.cross_sections[i].b
                    y_flip, y_2 = hyd.compute_flip_and_conjugate(flow, self.weir_length, self.P)
                    y_ts.append(float(y_t))
                    y_flips.append(float(y_flip))
                    y_2s.append(float(y_2))

            # convert those lists to strings and save them in columns in our data frame
            y_ts_string = str(y_ts)
            y_flips_string = str(y_flips)
            y_2s_string = str(y_2s)
            lhd_df.loc[lhd_df['ID'] == self.id, f'y_t_{i + 1}'] = y_ts_string
            lhd_df.loc[lhd_df['ID'] == self.id, f'y_flip_{i + 1}'] = y_flips_string
            lhd_df.loc[lhd_df['ID'] == self.id, f'y_2_{i + 1}'] = y_2s_string

        # update the csv file
        lhd_df.to_csv(lhd_csv, index=False)


    def set_dam_height(self, P):
        self.P = P


    def plot_rating_curves(self):
        for cross_section in self.cross_sections[:-1]:
            cross_section.create_rating_curve()
        plt.xlabel('Flow (m$^{3}$/s)')
        plt.ylabel('Depth (m)')
        plt.title(f'Rating Curves for LHD No. {self.id}')
        plt.legend(title="Rating Curve Equations", loc='best', fontsize='small')
        plt.show()
        plt_loc = os.path.join(self.fig_dir, f"Rating Curves for LHD No. {self.id}.png")
        plt.savefig(plt_loc)


    def plot_cross_sections(self):
        for cross_section in self.cross_sections:
            cross_section.plot_cross_section()


    def plot_all_curves(self):
        for cross_section in self.cross_sections[:-1]:
            cross_section.create_combined_fig()


    # noinspection PyTypeChecker
    def plot_map(self):
        # Step 1: Open raster and reproject to EPSG:3857
        with rasterio.open(self.tif_loc) as src:
            dst_crs = 'EPSG:3857'
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)

            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
            })

            memfile = MemoryFile()
            with memfile.open(**kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)
            reprojected = memfile.open()
            raster_data = reprojected.read(1)
            transform = reprojected.transform

        # Step 2: Load and reproject shapefile
        gdf = gpd.read_file(self.shp_loc)
        gdf = gdf.to_crs('EPSG:3857')

        # Step 3: Set zoom bounds
        buffer = 100  # meters
        minx, miny, maxx, maxy = gdf.total_bounds
        minx -= buffer * 2
        miny -= buffer
        maxx += buffer * 2
        maxy += buffer

        # Step 4: Plot
        # Plot setup
        fig, ax = plt.subplots(figsize=(10, 10))

        # Set plot extent early
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        # Step P-1: Add basemap (lowest layer)
        ctx.add_basemap(ax, crs='EPSG:3857', source=ctx.providers.Esri.WorldImagery, zorder=0)

        # Step P-2: Plot raster on top of basemap
        show(raster_data, transform=transform, ax=ax, cmap='plasma', zorder=1, alpha=0.7)

        # Step P-3: Plot shapefile points on top of both
        # separate into upstream and downstream cross-sections
        gdf_upstream = gdf.iloc[[-1]]
        gdf_downstream = gdf.iloc[:-1]
        gdf_upstream.plot(ax=ax, color='green', markersize=100, edgecolor='black', zorder=2, label="Upstream")
        gdf_downstream.plot(ax=ax, color='dodgerblue', markersize=100, edgecolor='black', zorder=2, label="Downstream")

        # Transformer from raster CRS (e.g., EPSG:3857) to EPSG:4326 (lon/lat)
        proj = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

        # Get current tick positions
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()

        # Convert to lon/lat
        xticks_lon = [proj.transform(x, yticks[0])[0] for x in xticks]
        yticks_lat = [proj.transform(xticks[0], y)[1] for y in yticks]

        # Set fixed ticks and labels
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.set_xticklabels([f"{lon:.4f}" for lon in xticks_lon])

        ax.yaxis.set_major_locator(FixedLocator(yticks))
        ax.set_yticklabels([f"{lat:.4f}" for lat in yticks_lat])

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Cross-Section Locations for LHD No. {self.id}")
        ax.legend(title="Cross-Section Location", title_fontsize="xx-large",
                   loc='upper right', fontsize='x-large')
        ax.set_axis_on()
        fig.tight_layout()
        return fig
