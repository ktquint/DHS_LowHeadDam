"""
What I asked Copilot:
Hey buddy... let's say that—hypothetically—I want to create a batch of maps using QGIS.
I have a bunch of directories with geopackages and shapefiles.
I want to go through each directory and create the same type of map for each.
Can I do this using python and avoid manually creating it in QGIS?? Thanks a million

What Copilot gave me:
"""

import os
import processing
from qgis.gui import QgsMapCanvas
from qgis.analysis import QgsNativeAlgorithms
from processing.core.Processing import Processing
from qgis.core import (QgsApplication, QgsProject, QgsVectorLayer)


# Initialize QGIS Application
QgsApplication.setPrefixPath("/path/to/qgis", True)
qgs = QgsApplication([], False)
qgs.initQgis()

# Set up processing framework
Processing.initialize()
QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

# Function to create maps
def create_map(directory):
    project = QgsProject.instance()
    project.clear()

    # Load geopackages and shapefiles
    for file in os.listdir(directory):
        if file.endswith(".gpkg") or file.endswith(".shp"):
            layer = QgsVectorLayer(os.path.join(directory, file), file, "ogr")
            if not layer.isValid():
                print(f"Layer {file} failed to load!")
            else:
                project.addMapLayer(layer)

    # Apply styling and export map
    # (Add your styling code here)

    # Export map
    # (Add your export code here)

# Loop through directories
base_dir = "/path/to/your/directories"
for subdir in os.listdir(base_dir):
    create_map(os.path.join(base_dir, subdir))

# Exit QGIS
qgs.exitQgis()