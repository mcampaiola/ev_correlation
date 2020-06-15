from elasticsearch import Elasticsearch
import json

HOST = "192.168.1.35"
PORT = 9200
pir_index_name = 'pir_sensor_new'
zoneminder_index_name = 'zoneminder_new'
index_name = 'zm_pir_new'

# Check elastic search connection
def connect_elasticsearch():
    es = None
    es = Elasticsearch([{'host': HOST, 'port': PORT}])
    if es.ping():
        print('Successfully connected')
    else:
        print('Connection failed!')
    return es

# Create index
def create_index(es, index_name):
    created = False
    # index settings
    settings = \
        {
            "aliases": {},
            "mappings": {
                "_meta": {},
                "properties":
                    {
                        "Date": {"type": "text"},
                        "Pir_Read_Value_1": {"type": "integer"},
                        "Pir_Read_Value_2": {"type": "integer"},
                        "Pir_Read_Value_3": {"type": "integer"},
                        "Microphone": {"type": "integer"},
                        #"Motion": {"type": "boolean"},
                        "AvgScore": {"type": "integer"},
                        "AlarmFrames": {"type": "integer"},
                        "Alert": {"type": "integer"},
                    }
            },
            "settings": {
                "index": {
                    "number_of_shards": "1",
                    "number_of_replicas": "1",
                }
            }
        }
    try:
        if not es.indices.exists(index_name):
            # Ignore 400 means to ignore "Index Already Exist" error.
            es.indices.create(index=index_name, body=settings)
            print('Created Index')
        created = True
    except Exception as ex:
        print(str(ex))
    finally:
        return created

if __name__ == '__main__':
    es = connect_elasticsearch()

    if es is not None:
        # Build query for pir events
        search_object_pir = {"size": 1000, "query": {"match_all": {}}}
        # Send request and get data back
        pir_res = es.search(index=pir_index_name, body=json.dumps(search_object_pir))
        pir_data = pir_res['hits']['hits']
        print(str(len(pir_data)) + " pir documents found")

        # Build query for zoneminder events
        search_object_pir = {"size": 200, "query": {"match_all": {}}}
        # Send request and get data back
        zm_res = es.search(index=zoneminder_index_name, body=json.dumps(search_object_pir))
        zm_data = zm_res['hits']['hits']
        print(str(len(zm_data)) + " zoneminder documents found")

        # Create zm_pir index
        create_index(es, index_name)

        # Index zm_pir ID: start form 0
        id = 0

        for pir_element in pir_data:
            pir_event = pir_element['_source']

            # Query body structure
            body = {
                'Date': pir_event['Date'],
                'Pir_Read_Value_1': pir_event['Read_Value_1'],
                'Pir_Read_Value_2': pir_event['Read_Value_2'],
                'Pir_Read_Value_3': pir_event['Read_Value_3'],
                'Microphone': pir_event['Microphone'],
                #'Motion': 'false',  # true/false
                'AvgScore': 0,  # 0/zm score
                'AlarmFrames': 0,  # 0/ n. of zm event frames
                'Alert': 0  # true/false
            }
            # This variable is intended to distinguish two case: a zm event containing the pir timestamp exists or does't exists
            found = False
            for zm_element in zm_data:
                zm_event = zm_element['_source']['Event']

                if (pir_event['Date'] >= zm_event['StartTime']) and (pir_event['Date'] < zm_event['EndTime']):

                    #body['Motion'] = 'true'
                    body['AvgScore'] = zm_event['AvgScore']
                    body['AlarmFrames'] = zm_event['AlarmFrames']

                    # Both the camera and pir have detected motion: set Alert to TRUE
                    if (pir_event['Read_Value_1'] == 1 or int(zm_event['AvgScore']) >= 7): #or zm_event['AvgScore'] >=
                        body['Alert'] = 1

                    # Only pir has detected motion: set Alert to FALSE
                    else:
                        body['Alert'] = 0

                    # Anyway, write the document in the index
                    res = es.index(index=index_name, id=id, body=body)
                    print ("Evento zm, ID: " + str(id))
                    id = id + 1
                    #print(res['result'])
                    found = True
                    break

            # Write the document in the index in case the pir event is not in any zm event
            if found == False:
                # Send document to elasticsearch index
                res = es.index(index=index_name, id=id, body=body)
                print("Evento pir, ID: " + str(id))
                id = id + 1
