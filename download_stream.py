import geopandas as gpd

# Path to the GDB file
gdb_path = "/Users/kennyquintana/Downloads/NHDPLUS_H_1602_HU4_20220412_GDB/NHDPLUS_H_1602_HU4_20220412_GDB.gdb"

# Name of the layer you want to extract
layer_name = "NHDFlowline"

# Read the layer from the GDB file
gdf = gpd.read_file(gdb_path, layer=layer_name)

# Output path for the extracted layer
output_path = "/Users/kennyquintana/Downloads/NHD_test.shp"

# Save the extracted layer to a shapefile
gdf.to_file(output_path)

print(f"Layer {layer_name} has been extracted to {output_path}")