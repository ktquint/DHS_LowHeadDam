import geopandas as gpd
import rasterio
from shapely.geometry import box

# Paths to the shapefile and DEM
shapefile_path = "/Users/kennyquintana/Downloads/NHD_test.shp"
dem_path = "/Users/kennyquintana/Downloads/USGS_one_meter_x44y447_UT_FEMAHQ_B2_QL1_2018.tif"

# Read the shapefile
gdf = gpd.read_file(shapefile_path)

# Open the DEM and get its CRS
with rasterio.open(dem_path) as dem:
    dem_crs = dem.crs
    dem_bounds = dem.bounds

# Reproject the shapefile to match the DEM's CRS if necessary
if gdf.crs != dem_crs:
    gdf = gdf.to_crs(dem_crs)

# Create a bounding box from the DEM bounds
dem_bbox = box(dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top)

# Trim the shapefile based on the DEM bounds
trimmed_gdf = gdf[gdf.intersects(dem_bbox)]

# Output path for the trimmed shapefile
output_path = "/Users/kennyquintana/Downloads/trimmed_shapefile.shp"

# Save the trimmed shapefile
trimmed_gdf.to_file(output_path)

print(f"Shapefile has been trimmed based on the DEM and saved to {output_path}")