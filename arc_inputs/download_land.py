import requests


def get_nlcd_data(lat, lon):
    # Define the URL for the NLCD API
    url = "https://www.mrlc.gov/geoserver/mrlc_display/NLCD_Land_Cover_L48/wms"

    # Define the parameters for the API request
    params = {
        'service': 'WMS',
        'version': '1.1.1',
        'request': 'GetMap',
        'layers': 'NLCD_Land_Cover_L48',
        'styles': '',
        'bbox': f"{lon - 0.01},{lat - 0.01},{lon + 0.01},{lat + 0.01}",
        'width': 256,
        'height': 256,
        'srs': 'EPSG:4326',
        'format': 'image/png',
        'transparent': 'true'
    }

    # Make the API request
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Save the image to a file
        with open('nlcd_data.png', 'wb') as file:
            file.write(response.content)
        print("NLCD data downloaded successfully.")
    else:
        print("Failed to download NLCD data.")


# Example usage
latitude = 40.2338
longitude = -111.6585
get_nlcd_data(latitude, longitude)