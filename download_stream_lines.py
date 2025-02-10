# Most recent, worked but really slow (2/7/25)
import geopandas as gpd
import os


def save_streamline_as_shapefile(gpkg_path, linkno, output_path):
    # Read the .gpkg file
    gdf = gpd.read_file(gpkg_path)

    # Filter the GeoDataFrame for the specified LINKNO
    streamline = gdf[gdf['LINKNO'] == linkno]

    if streamline.empty:
        print(f"No streamline found with LINKNO {linkno}")
        return

    # Save the filtered GeoDataFrame as a shapefile
    streamline.to_file(output_path, driver='ESRI Shapefile')
    print(f"Streamline data saved successfully to {output_path}")


# Example usage
gpkg_path = 'http://geoglows-v2.s3-us-west-2.amazonaws.com/streams/streams_106.gpkg'
linkno = 120034993  # Replace with your LINKNO value
output_path = 'data/streamline.shp'

save_streamline_as_shapefile(gpkg_path, linkno, output_path)

# This program takes a while to run from the one time I ran it; however, it seems to successfully save the streamline
# associated with the linkno as a .shp file

