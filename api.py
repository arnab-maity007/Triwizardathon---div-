import requests

api_key = 'RGAPI-91d4ffc1-0ffd-48f5-b6cc-a051f8df3135'
summoner_id = 'arkinkaansra1'
platform = 'sg2'

url = f'https://{platform}.api.riotgames.com/lol/spectator/v4/by-summoner/{summoner_id}'

headers = {
    'X-Riot-Token': api_key
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    game_data = response.json()
    print(game_data)
else:
    print(f'Error: {response.status_code} - {response.text}')
