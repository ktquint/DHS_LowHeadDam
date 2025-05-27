"""
This file holds all the objects we'll be using

CrossSections holds all the information we have associated with a given cross-section. The objects will be made using
the ID_XS_Out.txt file for each dam

Dam holds the cross-section information
"""
import os
import ast
import math

import geoglows
import numpy as np
import pandas as pd
import dbfread as dbf
from typing import Hashable
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

import pyproj
import rasterio
import geopandas as gpd
import contextily as ctx
from rasterio.plot import show
from rasterio.io import MemoryFile
from matplotlib.ticker import FixedLocator
from rasterio.warp import calculate_default_transform, reproject, Resampling


g = 9.81 # grav. const.


def eq_3(H, P, q):
    return (2 / 3) * (0.611 + 0.075 * (H / P)) * np.sqrt(2 * g) * H ** (3/2) - q


def round_sigfig(num, sig_figs):
    if num == 0:
        return 0
    else:
        return round(num, sig_figs - int(math.floor(math.log10(abs(num)))) - 1)


def x_intercept(x_1, y_1, y_2):
    for i in range(1, len(x_1)):
        y_low, y_high = y_1[i-1], y_1[i]
        x_low, x_high = x_1[i-1], x_1[i]

        if (y_low - y_2) * (y_high - y_2) <= 0 and y_low != y_high:
            # linearly interpolate
            ratio = (y_2 - y_low) / (y_high - y_low)
            x_2 = x_low + ratio * (x_high - x_low)
            return x_2
    return None


def get_streamflow(comid, date):
    """
    comid needs to be an int
    date needs to be in the format "%Y-%m-%d"

    returns average streamflow—for the entire record if no lat-long is given, else it's the average from the dates the lidar was taken
    """
    try:
        comid = int(comid)
    except ValueError:
        raise ValueError("comid needs to be an int")

    # this is all the data for the comid
    try:
        historic_df = geoglows.data.retrospective(river_id=comid, bias_corrected=True)
        historic_df.index = pd.to_datetime(historic_df.index)  # Ensure it's a DateTimeIndex

        if '-' in date and len(date) == 10:  # YYYY-MM-DD (corrected)
            date_dt = pd.to_datetime(date, format='%Y-%m-%d', errors="coerce")
            if pd.isna(date_dt):  # Check if conversion failed
                return None
            filtered_df = historic_df[historic_df.index.strftime('%Y-%m-%d') == date_dt.strftime('%Y-%m-%d')]

        elif '-' in date and len(date) == 7:  # MM/YYYY
            date_dt = pd.to_datetime(date, format='%Y-%m', errors="coerce")
            if pd.isna(date_dt):
                return None
            filtered_df = historic_df[(historic_df.index.year == date_dt.year) & (historic_df.index.month == date_dt.month)]

        elif len(date) == 4 and date.isdigit():  # YYYY
            date_dt = pd.to_datetime(date, format='%Y', errors="coerce")
            if pd.isna(date_dt):
                return None
            filtered_df = historic_df[historic_df.index.year == date_dt.year]


        else:
            return None

        if not filtered_df.empty:
            return filtered_df.median().values[0]

    except Exception as e:
        print(f"Error retrieving streamflow data for COMID {comid} and date {date}: {e}")

    return None


def dam_height(Q, b, delta_wse, y_t, delta_z=0):
    """
    eq.1
    P = delta_wse - H + y_t + dela_z
    ... but we need to find H
    also—all these calcs are in metric units

    eq.2
    q = (2/3) * C_w * np.sqrt(2 * g) * H**(3/2)
    where,
    C_w = 0.611 + 0.075 * H/P
    """

    # constant terms
    q = Q/b  # discharge (m^2/s)
    ## derived constants
    A = (2 / 3) * np.sqrt(2 * g)
    D = delta_wse + y_t + delta_z  # total pressure head + elevation

    # Function to solve
    def func(H):
        if H >= D:  # avoid division by zero or negative P
            return 1e6
        lhs = q / A
        rhs = 0.611 * H ** (3 / 2) + 0.075 * H ** (5 / 2) / (D - H)
        return lhs - rhs

    # Initial guess for H (must be less than D)
    H_0 = D * 0.8
    # Solve for H
    H_sol = fsolve(func, H_0)[0]

    # plug H into eq.1
    P = D - H_sol
    return round(P, 3)


def weir_height(Q, b, y_u, tol=0.001):
    """
    Q = flow in river (cms)
    b = bank width (m)
    y_u = upstream depth (m)
    """
    q = Q / b # unit flow
    # left-hand side
    A = 3 * q / (2 * np.sqrt(2 * g))
    # initial weir height estimate
    p = 0.5 * y_u # we want to start with a positive number < y_u
    # right-hand side
    B = 0.611 * (y_u - p)**(3/2) + 0.075 * ((y_u - p)**(5/2))/p
    counter = 0 # to avoid infinite loop
    while abs(A - B) > tol:
        counter += 1
        if A < B:
            p += 0.001
        else:
            p -= 0.001
        # recalculate B after adjusting height
        B = 0.611 * (y_u - p) ** (3 / 2) + 0.075 * ((y_u - p) ** (5 / 2)) / p
        if counter > 10000:
            break
    return round(p, 3)


class CrossSection:
    def __init__(self, index, xs_row, dam_id, weir_length, fig_dir):
        """
        lateral is a [] of lateral distances
        elevation is a [] of elevations
        wse is water surface elevation as a float
        distance is the downstream or upstream distance from the dam
        location is either 'Downstream' or 'Upstream'

        rating curve eq. d = a * Q **b
        depth_a is a & depth_b is b
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
            # upstream cross-section is 1/8 the weir length from the dam
            self.distance = int(self.L / 8)
        else:
            self.location = 'Downstream'
            # downstream cross-sections are one weir length farther than the previous
            self.distance = int((self.index + 1) * self.L)

        # rating curve info
        self.a = xs_row['depth_a']
        self.b = xs_row['depth_b']
        self.max_Q = xs_row['QMax']
        self.slope = round_sigfig(xs_row['slope'], 3)
        self.fatal_qs = None # np.array(ast.literal_eval(id_row['fatality_flows'].values[0]))

        # cross-section plot info
        y_1 = xs_row['elev_1']
        y_1 = y_1[::-1]
        y_2 = xs_row['elev_2']
        self.elevation = y_1 + y_2
        x_1 = [0 + j * xs_row['lat_1'] for j in range(len(y_1))]
        x_2 = [max(x_1) + j * xs_row['lat_2'] for j in range(len(y_2))]
        self.lateral = x_1 + x_2

        # water surface info
        self.wse = xs_row['wse_1']
        self.wse_x_1 = x_intercept(x_1, y_1, self.wse)
        self.wse_x_2 = x_intercept(x_2, y_2, self.wse)

        # initialize P, but don't give it a value yet
        self.P = None


    def set_dam_height(self, P):
        """
        pretty self-explanatory, no?
        """
        self.P = P


    def set_fatal_qs(self, q_list):
        self.fatal_qs = np.array(q_list)


    def plot_cross_section(self):
        # cross-section elevations
        plt.plot(self.lateral, self.elevation,
                 color='black', label=f'Downstream Slope: {self.slope}')
        # wse line
        wse_int = int(self.wse)
        plt.plot([self.wse_x_1, self.wse_x_2], [self.wse, self.wse],
                 color='cyan', linestyle='--', label=f'Water Surface Elevation: {wse_int} m')
        plt.xlabel('Lateral Distance (m)')
        plt.ylabel('Elevation (m)')
        plt.title(f'{self.location} Cross-Section {self.distance} meters from LHD No. {self.id}')
        plt.legend(loc='upper right')
        # the file name stands for Downstream/Upstream Cross-section No. XX at Low-Head Dam No. XX
        if self.index == 4:
            location = 'US'
            fig_loc = os.path.join(self.fig_dir, f"{location}_XS_LHD_{self.id}.png")
        else:
            location = 'DS'
            fig_loc = os.path.join(self.fig_dir, f"{location}_XS_{self.index}_LHD_{self.id}.png")
        plt.savefig(fig_loc)
        plt.show()


    def create_rating_curve(self):
        x = np.linspace(1, self.max_Q, 100)
        y = self.a * x ** self.b
        plt.plot(x, y,
                 label=f'Rating Curve {self.distance} meters {self.location}: $y = {self.a:.3f} x^{{{self.b:.3f}}}$')


    def plot_rating_curve(self):
        x = np.linspace(1, self.max_Q, 100)
        y = self.a * x ** self.b
        plt.plot(x, y, color='black', label=f'$y = {self.a:.3f} x^{{{self.b:.3f}}}$')

        plt.xlabel('Flow (m$^{3}$/s)')
        plt.ylabel('Depth (m)')
        plt.title(f'{self.location} Rating Curve {self.distance} meters from LHD No. {self.id}')
        plt.legend(title=f'{self.location} Rating Curve Equation')

        plt.grid(True)
        plt.show()


    def plot_flip_sequent(self):
        Q = np.linspace(1, self.max_Q, 100)
        Y_T = self.a * Q ** self.b
        depth_dict = {"Q": Q, "Y_T": Y_T}
        depth_df = pd.DataFrame(depth_dict)
        depth_df["L"] = self.L
        depth_df["P"] = self.P

        # hydraulic calcs
        depth_df["H"] = None
        for index, row in depth_df.iterrows():
            H_guess = 1.0
            P_i = row["P"]
            q_i = row["Q"] / row["L"]
            # if this doesn't work, go back to H_guess not in an array
            H_solution = fsolve(eq_3, np.array([H_guess]), args=(P_i, q_i))
            depth_df.at[index, "H"] = H_solution

        depth_df["H+P"] = depth_df["H"] + depth_df["P"]
        depth_df["H/(H+P)"] = depth_df["H"] / depth_df["H+P"]
        depth_df["C_W"] = 0.611 + 0.075 * (depth_df["H"] / depth_df["P"])
        depth_df["C_D"] = (2/3) * depth_df["C_W"] * np.sqrt(2 *g)
        depth_df["C_L"] = 0.1 * depth_df["P"] / depth_df["H"]

        # ideal jump calcs
        depth_df["Y_1/H"] = None
        index: Hashable
        for index, row in depth_df.iterrows():
            a = 1
            b = -(1 + row["P"] / row["H"])
            c = 0
            d = (4 / 9) * row["C_W"]**2 * (1 + row["C_L"])
            coeffs = [a, b, c, d]
            roots = np.roots(coeffs)

            positive_real_roots = [r.real for r in roots if np.isreal(r) and r.real > 0]
            try:
                depth_df.at[index, "Y_1/H"] = min(positive_real_roots)
            except ValueError:
                depth_df.at[index, "Y_1/H"] = 1
            # try:
            #     depth_df.at[index, "Y_1/H"] = min(positive_real_roots)
            # except ValueError: # if there are no real roots, use the last real value we found... only happens on one
            #     depth_df.at[index, "Y_1/H"] = depth_df.at[(index-1), "Y_1/H"]

        depth_df["Y_1"] = depth_df["Y_1/H"] * depth_df["H"]
        depth_df["V_1"] = depth_df["Q"] / depth_df["L"] / depth_df["Y_1"]
        depth_df["F_1"] = depth_df["V_1"] / g ** 0.5 / depth_df["Y_1"] ** 0.5
        depth_df["Y_2/H"] = depth_df["Y_1/H"] / 2 * (-1 + (1 + 8 * depth_df["F_1"] ** 2) ** 0.5)
        depth_df["Y_2"] = depth_df["Y_2/H"] * depth_df["H"]

        # depth_df["Y_T"] is already calculated
        depth_df["S"] = (depth_df["Y_T"] - depth_df["Y_2"]) / depth_df["Y_2"]

        # submerged hydraulic jump
        depth_df["Y_Flip"] = depth_df["H+P"] / 1.1

        # plot all 3 rating curves
        # plt.plot(depth_df["Q"], depth_df["Y_Flip"], label="Flip Depth")
        # plt.plot(depth_df["Q"], depth_df["Y_T"], label="Tailwater Depth")
        # plt.plot(depth_df["Q"], depth_df["Y_2"], label="Sequent Depth")
        # american units *eagle screech*
        plt.plot(depth_df["Q"]*35.315, depth_df["Y_Flip"]*3.281,
                 label="Flip Depth", color='gray', linestyle='--')
        plt.plot(depth_df["Q"]*35.315, depth_df["Y_T"]*3.281,
                 label="Tailwater Depth", color='dodgerblue', linestyle='-')
        plt.plot(depth_df["Q"]*35.315, depth_df["Y_2"]*3.281,
                 label="Sequent Depth", color='gray', linestyle='-')

        # make the plot look more presentable
        plt.grid(True)
        plt.xlim(left=0)
        plt.ylim(bottom=0)

        # add labels and title then show
        plt.xlabel('Discharge (ft$^{3}$/s)')
        plt.ylabel('Depth (ft)')
        # plt.xlabel('Discharge (m$^{3}$/s)')
        # plt.ylabel('Depth (m)')

        plt.title(f'Submerged Hydraulic Jumps at Low-Head Dam No. {self.id}')
        plt.legend(loc='upper left')


    def plot_fatal_flows(self):
        fatal_m = self.a * self.fatal_qs**self.b
        plt.scatter(self.fatal_qs * 35.315, fatal_m * 3.281,
                 label="Recorded Fatality", marker='o',
                 facecolors='none', edgecolors='black')


    def create_combined_fig(self):
        self.plot_flip_sequent()
        self.plot_fatal_flows()
        # the file name stands for Rating Curve No. XX at Low-Head Dam No. XX
        fig_loc = os.path.join(self.fig_dir, f"RC_{self.index+1}_LHD_{self.id}.png")
        plt.savefig(fig_loc)
        plt.show()




class Dam:
    """
    create a Dam based on the BYU LHD IDs
    add cross-sections with information from vdt & cross_section files
    """
    def __init__(self, lhd_id, lhd_csv, project_dir):
        #database information
        self.id = int(lhd_id)
        lhd_df = pd.read_csv(lhd_csv)

        # create a folder to store figures...
        results_dir = os.path.join(project_dir, "LHD_Results", str(self.id)) # we'll use this a lot later
        self.fig_dir = os.path.join(results_dir, "FIGS")
        os.makedirs(self.fig_dir, exist_ok=True)

        # add height columns before I extract the row
        P_guesses = ['P_1', 'P_2', 'P_3', 'P_4']
        slope_est = ['s_1', 's_2', 's_3', 's_4']
        for column in P_guesses:
            if column not in lhd_df.columns:
                lhd_df[column] = None
        for column in slope_est:
            if column not in lhd_df.columns:
                lhd_df[column] = None
        id_row = lhd_df[lhd_df['ID'] == self.id]

        # geographic information
        self.latitude = id_row['latitude'].values[0]
        self.longitude = id_row['longitude'].values[0]

        # physical information
        self.cross_sections = []
        self.weir_length = id_row['weir_length'].values[0]
        self.height = 0

        # fatality dates and fatal flows
        date_string = id_row['Date of Fatality'].iloc[0]
        dates = date_string.strip("[]").split(", ")

        formatted_dates = []

        for date in dates:
            try:
                # Try converting full date (MM/DD/YYYY → YYYY-MM-DD)
                formatted_date = pd.to_datetime(date.strip(), format="%m/%d/%Y").strftime("%Y-%m-%d")
            except ValueError:
                try:
                    # Try converting partial date (MM/YYYY → YYYY-MM)
                    formatted_date = pd.to_datetime(date.strip(), format="%m/%Y").strftime("%Y-%m")
                except ValueError:
                    # If neither format applies, keep original
                    formatted_date = date.strip()

            formatted_dates.append(formatted_date)

        self.fatality_dates = formatted_dates

        fatal_flows = []
        for date in self.fatality_dates:
            flow_value = get_streamflow(id_row['LINKNO'].iloc[0], date)  # Ensure single value extraction
            # print(flow_value)
            fatal_flows.append(float(flow_value))  # Convert to standard float

        self.fatal_flows = fatal_flows

            # find attributes based on the vdt and xs files
        vdt_loc = os.path.join(results_dir, "VDT", f"{str(self.id)}_Local_CurveFile.dbf")
        xs_loc = os.path.join(results_dir, "XS", f"{str(self.id)}_XS_Out.txt" )

        # save tif and shp files for later...
        self.shp_loc = os.path.join(results_dir, "VDT", f"{str(self.id)}_Local_CurveFile.shp")
        self.tif_loc = os.path.join(results_dir, "Bathymetry", f"{str(self.id)}_ARC_Bathy.tif")

        # read in dbf then translate it to data.frame
        vdt_dbf = dbf.DBF(vdt_loc)
        vdt_df = pd.DataFrame(iter(vdt_dbf))

        # read in txt as data.frame
        xs_df = pd.read_csv(xs_loc, header=None, sep='\t')
        xs_df.rename(columns={0: 'COMID', 1: 'Row', 2: 'Col', 3: 'elev_1',
                              4: 'wse_1', 5: 'lat_1', 6: 'n_1', 7: 'elev_2',
                              8: 'wse_2', 9: 'lat_2', 10: 'n_2', 11: 'slope'},
                     inplace=True)

        # evaluate the strings as literals (lists)
        xs_df['elev_1'] = xs_df['elev_1'].apply(ast.literal_eval)
        xs_df['n_1'] = xs_df['n_1'].apply(ast.literal_eval)
        xs_df['elev_2'] = xs_df['elev_2'].apply(ast.literal_eval)
        xs_df['n_2'] = xs_df['n_2'].apply(ast.literal_eval)

        # let's merge the tables (how='left' because vdt only contains the xs we want and xs has all of them)
        merged_df = pd.merge(vdt_df, xs_df, on=['COMID', 'Row', 'Col'], how='left')

        # let's go through each row of the df and create cross-sections objects
        for index, row in merged_df.iterrows():
            self.cross_sections.append(CrossSection(index, row, self.id, self.weir_length, self.fig_dir))

        # hydrologic information
        self.known_baseflow = id_row['known_baseflow'].values[0]

        # let's add the dam height and slope to the csv
        for i in range(len(self.cross_sections)-1):
            Q_i = self.known_baseflow
            L_i = self.weir_length
            delta_wse_i = self.cross_sections[-1].wse - self.cross_sections[i].wse

            # tailwater using the power function
            # y_i = self.cross_sections[i].a * Q_i**self.cross_sections[i].b
            # tailwater using the wse and depth
            y_i = self.cross_sections[i].wse - min(self.cross_sections[i].elevation)

            # estimate dam height, add to cross-section and csv file
            z_i = -1 * self.cross_sections[i].slope * self.cross_sections[i].distance
            P_i = dam_height(Q_i, L_i, delta_wse_i, y_i, z_i)
            # ths one has delta z = 0
            # P_i = dam_height(Q_i, L_i, delta_wse_i, y_i)

            self.cross_sections[i].set_dam_height(P_i)
            lhd_df.loc[lhd_df['ID'] == self.id, f'P_{i + 1}'] = P_i * 3.281 # convert to ft

            # add fatal qs to each cross-section
            self.cross_sections[i].set_fatal_qs(self.fatal_flows)

            # add slope info to csv file
            s_i = self.cross_sections[i].slope
            lhd_df.loc[lhd_df['ID'] == self.id, f's_{i + 1}'] = s_i

        lhd_df['fatal flows'] = None
        lhd_df.loc[lhd_df['ID'] == self.id, 'fatal flows'] = str(self.fatal_flows)

        # update the csv file
        lhd_df.to_csv(lhd_csv, index=False)


    def set_dam_height(self, P):
        self.height = P


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
        plt.title(f"Cross-Section Locations for LHD No. {self.id}")
        plt.legend(title="Cross-Section Location", title_fontsize="xx-large",
                   loc='upper right', fontsize='x-large')
        plt.axis('on')
        plt.tight_layout()
        plt.show()
