import json
from random import randrange
from datetime import datetime, timedelta
import requests
from elasticsearch import Elasticsearch
es = Elasticsearch(['http://192.168.1.35:9200'])

dict_events = {}
date_randoms = []
zm_file = open("zm_events.txt", "r")

lista_letture_zm = zm_file.readlines()
zm_file.close()

for i in range(0, len(lista_letture_zm)):
    dict_events[i] = lista_letture_zm[i].split(" ")
    #print(dict_events[i][0])
    date_randoms.append(datetime.strptime(dict_events[i][0], '%Y-%m-%dT%H:%M:%S') + timedelta(seconds=randrange(int(dict_events[i][2]) + 1)))
    #print(date_randoms[i])

# dizionario costituito da un unica entry('events':[lista vuota]);
# la lista andrà riempita con un dizionario a due entry
# {
#   'Date': str(datetime)
#   'Read_value': str(valore letto, 0 o 1)
# }
dict_pir = {'events' : []}
pir_file = open("pir_data.txt", "r")
lista_letture_pir = pir_file.readlines()
for i in range(0, len(lista_letture_pir)):
    dict_pir['events'].append({'Date': str(date_randoms[i])[0:10] + "T" + str(date_randoms[i])[11:19] + "+02:00", 'Read_Value': int(lista_letture_pir[i])})

print(dict_pir)

id = 0
for entry in dict_pir['events']:
    res = es.index(index="pir_sensor", id=id, body=entry)
    id = id + 1
    print(res['result'])
