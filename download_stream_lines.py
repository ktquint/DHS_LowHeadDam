import geopandas as gpd
import requests
from shapely.geometry import mapping

def download_gpkg(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

def find_stream_in_gpkg(gpkg_file, linkno):
    gdf = gpd.read_file(gpkg_file)
    stream = gdf[gdf['LINKNO'] == linkno]
    return stream

def save_stream_as_shapefile(stream, filename):
    stream.to_file(filename, driver='ESRI Shapefile')

# URL of the GeoPackage file
gpkg_url = "http://geoglows-v2.s3-website-us-west-2.amazonaws.com/streams.gpkg"

# Download the GeoPackage file
gpkg_filename = "streams.gpkg"
download_gpkg(gpkg_url, gpkg_filename)

# Specify the LINKNO value
linkno_value = 123456  # Replace with the actual LINKNO value

# Find the stream with the specified LINKNO value
stream = find_stream_in_gpkg(gpkg_filename, linkno_value)

# Save the stream as a shapefile
shapefile_filename = "stream.shp"
save_stream_as_shapefile(stream, shapefile_filename)

print(f"The stream with LINKNO {linkno_value} has been saved as {shapefile_filename}.")