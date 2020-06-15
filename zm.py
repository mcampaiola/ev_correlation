import json
import requests
from datetime import datetime, timedelta
import time
from elasticsearch import Elasticsearch
from signal import signal, SIGINT
from sys import exit

# CTRL + C handler
def handler(signal_received, frame):
    print('\nSIGINT or CTRL-C detected. Exiting gracefully')
    exit(0)

# Elasticsearch connector
es = Elasticsearch(['http://192.168.1.35:9200'])
INTERVAL = 2000
ZM_INDEX = "zoneminder_new"

if __name__ == '__main__':

    # Handle SIGINT event
    signal(SIGINT, handler)

    epoch = datetime.utcfromtimestamp(0)
    # Do the follow instructions every INTERVAL seconds

    # Get current time and past time (INTERVAL seconds before)
    current_time = datetime.now()
    past_time = current_time - timedelta(seconds=INTERVAL)

    # Set document ID as datetime in millisecons
    #id = int((current_time - epoch).total_seconds() * 1000000) not working

    id = 0

    # Convert datetime objects in the required format
    end_time = str(datetime.date(current_time)) + "%20" + datetime.strftime(current_time, '%H:%M:%S')
    start_time = str(datetime.date(past_time)) + "%20" + datetime.strftime(past_time, '%H:%M:%S')

    # Camera ID
    monitor_id = 2

    # Build url request (for a specific monitor within a specific time interval)
    api_time_string = "/StartTime%20>=:" + start_time + "/EndTime%20<=:" + end_time
    base_url_api = "https://giushome.homepc.it:7443/zm/api/events/index/MonitorId:" + str(monitor_id) + api_time_string + ".json?token="

    # Read token saved in a file
    file_token = open("token.txt", "w+")
    token = file_token.readline()

    # Append token to base_url_api
    complete_url_api = base_url_api + token

    # Try to call api GET method with the saved token to get events
    resp = requests.get(complete_url_api)

    # Generate a new token if the saved one is expired
    if resp.status_code == 401:

        # Call api POST method to get new token
        request_token_url = "https://giushome.homepc.it:7443/zm/api/host/login.json"
        params = {'user': 'admin', 'pass': 'zm2020!'}
        resp_get_token = requests.post(request_token_url, params)
        json_resp = json.loads(resp_get_token.text)
        token = json_resp['access_token']

        # Append new token to base_url_api
        complete_url_api = base_url_api + token

        # Try to call api GET method with the new token to get events
        resp = requests.get(complete_url_api)
        print(complete_url_api)

        # Save new token in the same file which you read the old one from
        file_token.write(token)
        file_token.close()

    if resp.status_code != 200:
        # This means something went wrong.
        raise Exception('GET /tasks/ {}'.format(resp.status_code))

    # Store json response
    data = json.loads(resp.text)
    print(len(data['events']))

    # Add Zoneminder events in elasticsearch index
    for event in data['events']:

        # Convert time into required format (according to elasticsearch)
        event['Event']['StartTime'] = event['Event']['StartTime'][0:10] + "T" + event['Event']['StartTime'][11:19] + "+02:00"
        event['Event']['EndTime'] = event['Event']['EndTime'][0:10] + "T" + event['Event']['EndTime'][11:19] + "+02:00"
        # Put document in elasticsearch index
        res = es.index(index=ZM_INDEX, id=id, body=event)
        id = id + 1
        print(res['result'])
