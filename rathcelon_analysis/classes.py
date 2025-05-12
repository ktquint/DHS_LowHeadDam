"""
This file holds all the objects we'll be using

CrossSections holds all the information we have associated with a given cross-section. The objects will be made using
the ID_XS_Out.txt file for each dam

Dam holds the cross-section information
"""
import ast
import math
import geoglows
import numpy as np
import pandas as pd
import dbfread as dbf
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import requests
import re

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


def get_dem_dates(lat, lon):
    """
    Use lat/lon to get Lidar data used to make the DEM.
    Check the date the Lidar was taken.
    """
    bbox = (lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001)
    dataset = "Lidar Point Cloud (LPC)"
    base_url = "https://tnmaccess.nationalmap.gov/api/v1/products"

    params = {
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "datasets": dataset,
        "max": 1,
        "outputFormat": "JSON"
    }

    response = requests.get(base_url, params=params)
    lidar_info = response.json().get("items", [])

    if not lidar_info:
        print("No Lidar data found for the given coordinates.")
        return "No Lidar data found for the given coordinates."

    meta_url = lidar_info[0].get('metaUrl')
    if not meta_url:
        print("metaUrl key not found in the response.")
        return "metaUrl key not found in the response."

    response2 = requests.get(meta_url)
    html_content = response2.text

    match_start = re.search(r'<dt>Start Date</dt>\s*<dd>(.*?)</dd>', html_content, re.IGNORECASE)
    match_end = re.search(r'<dt>End Date</dt>\s*<dd>(.*?)</dd>', html_content, re.IGNORECASE)

    if match_start and match_end:
        start_date_value = match_start.group(1).strip()
        end_date_value = match_end.group(1).strip()
        return [start_date_value, end_date_value]
    else:
        print("Date parameters not found.")
        return "Date parameters not found."


def get_streamflow(comid, lat=None, lon=None):
    """
    comid needs to be an int
    date needs to be in the format %Y-%m-%d

    returns average streamflow—for the entire record if no lat-long is given, else it's the average from the dates the lidar was taken
    """
    try:
        comid = int(comid)
    except ValueError:
        raise ValueError("comid needs to be an int")

    # this is all the data for the comid
    historic_df = geoglows.data.retro_daily(comid)
    historic_df.index = pd.to_datetime(historic_df.index)

    if lat and lon is not None:
        try:
            date_range = get_dem_dates(lat, lon)
            subset_df = historic_df.loc[date_range[0]:date_range[1]]
            Q = np.median(subset_df[comid])
        except IndexError:
            date_range = get_dem_dates(lat, lon)
            raise ValueError(f"No data available for {date_range}")
    else:
        Q = np.median(historic_df[comid])

    return Q


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
    P = delta_wse - H_sol + y_t + delta_z
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
    def __init__(self, index, xs_row, id_row):
        """
        lateral is a [] of lateral distances
        elevation is a [] of elevations
        wse is water surface elevation as a float
        distance is the downstream or upstream distance from the dam
        location is either 'Downstream' or 'Upstream'

        rating curve eq. d = a * Q **b
        depth_a is a & depth_b is b
        """
        self.id = id_row['ID'].values[0]
        # geospatial info
        self.lat = xs_row['Lat']
        self.lon = xs_row['Lon']
        if index == 4:
            self.location = 'Upstream'
        else:
            self.location = 'Downstream'
        # rating curve info
        self.a = xs_row['depth_a']
        self.b = xs_row['depth_b']
        self.max_Q = xs_row['QMax']
        self.slope = round_sigfig(xs_row['slope'], 3)
        self.L = id_row['weir_length']
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
        # fix this later...
        self.distance = 100
        self.P = ""


    def set_dam_height(self, P):
        self.P = P


    def plot_cross_section(self):
        # cross-section elevations
        plt.plot(self.lateral, self.elevation,
                 color='black', label=f'Downstream Slope: {self.slope}')
        # wse line
        plt.plot([self.wse_x_1, self.wse_x_2], [self.wse, self.wse], color='cyan', linestyle='--')
        plt.xlabel('Lateral Distance (m)')
        plt.ylabel('Elevation (m)')
        plt.title(f'{self.location} Cross-Section {self.distance} meters from LHD No. {self.id}')
        plt.legend(loc='upper left')
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
        # Ensure self.L is a single value or access the first element if it's a Series
        if isinstance(self.L, pd.Series):
            depth_df["L"] = float(self.L.iloc[0])
        else:
            depth_df["L"] = float(self.L)
        depth_df["P"] = self.P

        # hydraulic calcs
        depth_df["H"] = ""
        for index, row in depth_df.iterrows():
            H_guess = 1.0
            P_i = row["P"]
            q_i = row["Q"] / row["L"]
            H_solution = fsolve(eq_3, H_guess, args=(P_i, q_i))
            depth_df.at[index, "H"] = H_solution

        depth_df["H+P"] = depth_df["H"] + depth_df["P"]
        depth_df["H/(H+P)"] = depth_df["H"] / depth_df["H+P"]
        depth_df["C_W"] = 0.611 + 0.075 * (depth_df["H"] / depth_df["P"])
        depth_df["C_D"] = (2/3) * depth_df["C_W"] * np.sqrt(2 *g)
        depth_df["C_L"] = 0.1 * depth_df["P"] / depth_df["H"]

        # ideal jump calcs
        depth_df["Y_1/H"] = ""
        for index, row in depth_df.iterrows():
            a = 1
            b = -(1 + row["P"] / row["H"])
            c = 0
            d = (4 / 9) * row["C_W"] * (1 + row["C_L"])
            coeffs = [a, b, c, d]
            roots = np.roots(coeffs)

            positive_real_roots = [r.real for r in roots if np.isreal(r) and r.real > 0]
            try:
                depth_df.at[index, "Y_1/H"] = min(positive_real_roots)
            except ValueError: # if there are no real roots, use the last real value we found... only happens on one
                depth_df.at[index, "Y_1/H"] = depth_df.at[index-1, "Y_1/H"]

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
        plt.show()


    def plot_fatal_flow(self):
        plt.plot()
        return


class Dam:
    """
    create a Dam based on the BYU LHD IDs
    add cross-sections with information from vdt & cross_section files
    """
    def __init__(self, lhd_id, lhd_csv, project_dir):
        #database information
        self.id = lhd_id
        lhd_df = pd.read_csv(lhd_csv)
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

        # self.top_width = 0
        self.height = 0
        self.h_overtop = []
        # find attributes based on the vdt and xs files
        vdt_loc = f'{project_dir}/LHD_Results/{self.id}/VDT/{self.id}_Local_CurveFile.dbf'
        xs_loc = f'{project_dir}/LHD_Results/{self.id}/XS/{self.id}_XS_Out.txt'
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
        self.max_Q = max(merged_df['QMax'].values)
        # let's go through each row of the df and add cross-sections to the dam.
        for index, row in merged_df.iterrows():
            self.cross_sections.append(CrossSection(index, row, id_row))

        # # hydrologic information
        self.known_baseflow = id_row['known_baseflow'].values[0]
        # fatality flow will be added soon...

        # let's add the dam height and slope to the csv
        for i in range(len(self.cross_sections)-1):
            Q_i = self.known_baseflow
            L_i = self.weir_length
            y_i = self.cross_sections[i].a * Q_i**self.cross_sections[i].b
            delta_wse_i = self.cross_sections[-1].wse - self.cross_sections[i].wse

            # energy_up = self.cross_sections[-1].wse - min(self.cross_sections[i].elevation)
            P_i = dam_height(Q_i, L_i, delta_wse_i, y_i)
            self.cross_sections[i].set_dam_height(P_i)
            lhd_df.loc[lhd_df['ID'] == self.id, f'P_{i + 1}'] = P_i * 3.281 # convert to ft
            s_i = self.cross_sections[i].slope
            lhd_df.loc[lhd_df['ID'] == self.id, f's_{i + 1}'] = s_i

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


    def plot_cross_sections(self):
        for cross_section in self.cross_sections:
            cross_section.plot_cross_section()

    def plot_all_curves(self):
        for cross_section in self.cross_sections[:-1]:
            cross_section.plot_flip_sequent()