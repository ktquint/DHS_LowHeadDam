import dbfread
import pandas as pd
import requests


# Custom parser that skips corrupt numeric fields
class SafeFieldParser(dbfread.FieldParser):
    def parseN(self, field, field_data):
        try:
            return super().parseN(field, field_data)
        except ValueError:
            return None  # Or float('nan') if you prefer NaN

# Read DBF with safe parser
table = dbfread.DBF(
    "C:/Users/ki87ujmn/Downloads/SWOT_Hotchkiss-20250703T163614Z-1-001/SWOT_Hotchkiss/LHD_SWOT_Locations.dbf",
    parserclass=SafeFieldParser
)

# Load into DataFrame
swot_df = pd.DataFrame(iter(table))

swot_df = swot_df.dropna(subset=['SWOT_ID'])
swot_df = swot_df.reset_index(drop=True)

# everything above this just finds the SWOT_ID for the LHDs

# hopefully never have to use this again lol
# def reformat_date(raw_date_str):
#     # Convert string to list
#     date_list = raw_date_str.strip("[]").split(",")
#     quoted_dates = [f'"{d.strip()}"' for d in date_list]
#     date_str_fixed = "[" + ", ".join(quoted_dates) + "]"
#
#     dates = ast.literal_eval(date_str_fixed)
#
#     formatted_dates = []
#     for date in dates:
#         date = date.strip()
#         try:
#             # Try parsing full date
#             dt = datetime.strptime(date, "%m/%d/%Y")
#             formatted_dates.append(f'"{dt.strftime("%Y-%m-%d")}"')
#         except ValueError:
#             # Skip dates that can't be parsed as full dates
#             continue
#
#     # Convert back to string that looks like a list
#     result_str = "[" + ", ".join(formatted_dates) + "]"
#     return result_str



lhd_df = pd.read_csv("E:\LHD_1-m_NWM\LowHead_Dam_Database.csv")

lhd_df = lhd_df.merge(swot_df[['ID', 'SWOT_ID']], on='ID', how='inner')
lhd_df['SWOT_ID'] = lhd_df['SWOT_ID'].astype('Int64')  # Allows NA values, too


# Q: i read in the database, now what?
# A: let's merge this dataframe with the one we made earlier, so we have dates and swot id's


for index, row in lhd_df.iterrows():
    fatality_dates = row['fatality_dates']
    swot_id = row['SWOT_ID']
    lat = row['latitude']
    lon = row['longitude']
    # date_list = db.get_dem_dates(lat, lon)
    # # Q: now i have a list of the dates... what do i do next?
    # # A: let's loop through the dates
    #
    # start_time = date_list[0]
    # end_time = date_list[1]

    start_time = "2025-01-01"
    end_time = "2025-01-31"

    swot_url = (
        f"https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries?"
        f"feature=Reach&feature_id={swot_id}&start_time={start_time}T00:00:00Z"
        f"&end_time={end_time}T00:00:00Z&output=geojson&fields=wse,time_str&compact=true"
    )
    response = requests.get(swot_url)

    # Check status
    if response.status_code == 200:
        data = response.json()  # Parse as JSON (GeoJSON in this case)
        print(data)
    else:
        print(f"Request failed with status code {response.status_code}")

    features = response.json()["results"]["geojson"]["features"]

    for feature in features:
        print(feature["properties"])
