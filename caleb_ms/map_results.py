import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point

shj_xlsx = "C:/Users/ki87ujmn/Downloads/SHJ_National_Incident.xlsx"
results_sheet = "ID_Chart_3"
shj_df = pd.read_excel(shj_xlsx, sheet_name=results_sheet).iloc[:, :12]
shj_df["Color"] = None
shj_df["Label"] = None

for index, row in shj_df.iterrows():
    if row["Conjugate Depth"] < row['Tailwater Depth'] < row['Flip Depth']:
        shj_df.loc[index, 'Color'] = 'red'
        shj_df.loc[index, 'Label'] = "Predicted as Type C"
    elif row['Conjugate Depth'] >= row['Tailwater Depth']:
        shj_df.loc[index, 'Color'] = 'blue'
        shj_df.loc[index, 'Label'] = "Predicted as Type A"
    else:
        shj_df.loc[index, 'Color'] = 'lime'
        shj_df.loc[index, 'Label'] = "Predicted as Type D"

# Convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(shj_df['Longitude'], shj_df['Latitude'])]
gdf = gpd.GeoDataFrame(shj_df, geometry=geometry, crs="EPSG:4326")

# Convert to Web Mercator for basemap
gdf = gdf.to_crs(epsg=3857)

# Plot
fig, ax = plt.subplots(figsize=(8.5, 11))
# Plot red first
for _, group in gdf[gdf['Color'] == 'red'].groupby(['Color', 'Label']):
    color = group['Color'].iloc[0]
    label = group['Label'].iloc[0]
    group.plot(ax=ax, color=color, label=label)

# Then plot everything else
for _, group in gdf[gdf['Color'] != 'red'].groupby(['Color', 'Label']):
    color = group['Color'].iloc[0]
    label = group['Label'].iloc[0]
    group.plot(ax=ax, color=color, label=label)

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)


# Clean up
ax.set_axis_off()
plt.title("Map of Low-Head Dams \n with Predicted Submerged Hydraulic Jumps",
          fontsize=16, fontweight='bold')
plt.legend(title="Legend", title_fontsize=12)
plt.show()