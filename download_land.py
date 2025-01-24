import requests


def get_nlcd_data(lat, lon):
    # Replace 'YOUR_API_KEY' with your actual API key
    api_key = 'YOUR_API_KEY'
    url = f'https://api.example.com/nlcd?lat={lat}&lon={lon}&key={api_key}'

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}")
        return None


# Example usage
latitude = 40.7128
longitude = -74.0060
nlcd_data = get_nlcd_data(latitude, longitude)

if nlcd_data:
    print(nlcd_data)
