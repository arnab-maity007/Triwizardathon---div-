import requests

# Replace with your actual API key
api_key = 'RGAPI-91d4ffc1-0ffd-48f5-b6cc-a051f8df3135'

# Replace with the summoner's Riot Game Name and Tagline
game_name = 'arkinkaansra1'  # Example Riot Game Name
tagline = '1470'  # Example Tagline

# Set the platform, e.g., sg2 for Southeast Asia
platform = 'sg2'

# Construct the URL to fetch the account details using Riot ID
url = f'https://{platform}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{game_name}/{tagline}'

# Set up headers with the API key
headers = {
    'X-Riot-Token': api_key
}

# Send the GET request to the Riot API
response = requests.get(url, headers=headers)

# Check if the response is successful
if response.status_code == 200:
    account_data = response.json()
    # Extract the encrypted summoner ID from the response
    encrypted_summoner_id = account_data['id']
    print(f"Encrypted Summoner ID: {encrypted_summoner_id}")
else:
    print(f"Error: {response.status_code} - {response.text}")
