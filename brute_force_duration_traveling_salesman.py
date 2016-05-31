import itertools
from pylab import *
import csv
import googlemaps
import responses
import pickle
import os.path
import cPickle
import googlemaps
from datetime import datetime


# BRUTE FORCE ALGORITHM
# Optimize route based on duration metrics
# Starting and ending points of the route are constant
# Route optimized between starting and ending points


## returns gmaps cost based on distances and durations
def get_gmaps_cost(origins, destinations, mode):
    matrix = gmaps.distance_matrix(origins, destinations, mode = mode)
    api_distance_matrix = (matrix[u'rows'][0][u'elements'][0][u'distance'][u'text'])
    api_distance_splitted = api_distance_matrix.split(" ")
    api_distance = float(api_distance_splitted[0])
    api_duration_matrix = (matrix[u'rows'][0][u'elements'][0][u'duration'][u'text'])
    api_duration_splitted = api_duration_matrix.split(" ")
    api_duration = float(api_duration_splitted[0])

    return api_distance, api_duration


## returns distance and duration between two locations
def distance(p1, p2, mode, my_distances, my_durations):

    distance_key = str(p1) + '_' + str(p2)

    if distance_key not in my_distances:
        origins = [[p1[0],p1[1]]]
        destinations = [(p2[0],p2[1])]
        api_distance, api_duration = get_gmaps_cost(origins, destinations, mode)
        my_distances[distance_key] = api_distance
        my_durations[distance_key] = api_duration

    else:
        api_distance = my_distances[distance_key]
        api_duration = my_durations[distance_key]


    return api_distance, api_duration




## returns travel cost and duration
def calCosts(routes, nodes, mode, my_distances, my_durations):
    travelCosts_distances = []
    travelCosts_durations = []

    for route in routes:
        travelCost_distance = 0
        travelCost_duration = 0

        #Sums up the travel cost
        for i in range(1,len(route)):
            #takes an element of route, uses it to find the corresponding coords and calculates the distanc

            distance_cost, duration_cost = distance(nodes[str(route[i-1])], nodes[str(route[i])], mode, my_distances, my_durations)
            travelCost_distance += distance_cost
            duration_cost += 20
            travelCost_duration += duration_cost

        travelCosts_distances.append(travelCost_distance)
        travelCosts_durations.append(travelCost_duration)


    #pulls out the smallest travel cost
    smallestCost_distances = min(travelCosts_distances)
    smallestCost_durations = min(travelCosts_durations)


    shortest_distances = (routes[travelCosts_distances.index(smallestCost_distances)], smallestCost_distances)
    shortest_durations = (routes[travelCosts_durations.index(smallestCost_durations)], smallestCost_durations)

    #returns tuple of the route and its cost

    return shortest_distances, shortest_durations


## generates route
def genRoutes(routeLength):
    #lang hold all the 'alphabet' of nodes
    lang = [ x for x in range(2,routeLength+1) ]

    #uses built-in itertools to generate permutations
    routes = list(map(list, itertools.permutations(lang))
                  )
    #inserts the home city, must be the first city in every route
    for x in routes:
        x.insert(0,1)

    return routes


def main_route (nodes=None, instanceSize = 9, travel_mode = "driving"):

    if os.path.isfile("distance_" + travel_mode +".pkl") :                  # check if file exists
        with open("distance_" + travel_mode + ".pkl", "rb") as input_file1:
            my_distances = cPickle.load(input_file1)
    else:
        my_distances = {}

    if os.path.isfile("duration_" + travel_mode + ".pkl"):
        with open("duration_" + travel_mode + ".pkl", "rb") as input_file2:
            my_durations = cPickle.load(input_file2)
    else:
        my_durations = {}


    routes = genRoutes(instanceSize)
    shortest_distances, shortest_durations = calCosts(routes, Nodes, travel_mode, my_distances, my_durations)



    final_str = "var places = ["

    for item in shortest_durations[0]:
        final_str  += "{lat : " + str(Nodes[str(item)][0]) + ", lng:" + str(Nodes[str(item)][1]) + "},"

    final_str = final_str[:-1] + "]"


    with open(str(hour) + "hour_" + travel_mode + "_output.js", mode = "w") as f:
        f.write(final_str)


    # print("Shortest Route by Distance" + " by " + travel_mode + " :", shortest_distances[0])
    # print("Travel Cost by Distance" + " by " + travel_mode + ":", shortest_distances[1])
    print("Shortest Route by Duration"+ " by " + travel_mode + ":", shortest_durations[0])
    print("Travel Cost by Duration" + " by " + travel_mode + ":", shortest_durations[1])


    with open(r"distance_" + travel_mode + ".pkl", "wb") as output_file1:
        cPickle.dump(my_distances, output_file1)

    with open(r"duration_" + travel_mode + ".pkl", "wb") as output_file2:
        cPickle.dump(my_durations, output_file2)



if __name__=='__main__':


    filename = '/home/stm/PycharmProjects/isttouristic/ist_data_short.csv'

    positions = []
    id = []
    lats = []
    lons = []
    positions = []
    places = []
    types = []
    locations = []
    ids = []

    with open(filename) as file:
        file.next()
        train_reader = csv.reader(file, delimiter=',')


        for rows in train_reader:
            _id = int(rows[0])
            place = (rows[1])
            lat = float(rows[2])
            lon = float(rows[3])
            type = rows[4]

            id.append(_id)
            lats.append(lat)
            lons.append(lon)
            places.append(place)
            types.append(type)
            ids.append(_id)

            pos = {"place": place  ,"lat": float(lat), "lon": float(lon), "type": type}
            positions.append(pos)

            location = [lat,lon]
            locations.append(location)

    gmaps = googlemaps.Client(key='AIzaSyDLJRln6EUG180Lm6qqNQ15ZtdV-6SR8u4', timeout=None, connect_timeout=None, read_timeout=None, retry_timeout=1000)

    responses.add(responses.GET,
                  'https://maps.googleapis.com/maps/api/distancematrix/json',
                  body='{"status":"OK","rows":[]}',
                  status=200,
                  content_type='application/json')


    # starts from topkapi
    # Nodes = {
    #     '1': (41.010501,28.982369),
    #     '2': (41.015892, 28.977266),
    #     '3': (41.008583,28.980175),
    #     '4': (41.010501,28.982369),
    #     '5': (41.013485,28.980738),
    #     '6': (41.00893,28.977414),
    #     '7': (41.008533,28.971266),
    #     '8': (41.011065,28.96835),
    #     '9': (41.008238,28.974359),
    #     '10': (41.008236,28.976359)
    #
    # }



    Nodes = {
        '1': (41.010501,28.982369),
        '2': (41.0053,28.9768),
        '3': (41.008583,28.980175),
        '4': (41.013485,28.980738),
        '5': (41.008607,28.977848),
        '6': (41.008533,28.971266),
        '7': (41.011065,28.96835),
        '8': (41.0107,28.9681),
        '9': (41.016516,28.970455)

    }

    travel_mode = "walking"

    hour = 4
    k = 9

    main_route(Nodes, k, travel_mode)

    hour = 3
    k = 7
    main_route(Nodes, k, travel_mode)

    hour = 2.5
    k = 5
    main_route(Nodes, k, travel_mode)

    hour = 1.5
    k = 4
    main_route(Nodes, k, travel_mode)

    hour = 1
    k = 3
    main_route(Nodes, k, travel_mode)



    # travel_mode = "driving"
    #
    # hour = 3.5
    # k = 9
    #
    # main_route(Nodes, k, travel_mode)
    #
    # hour = 2.5
    # k = 7
    # main_route(Nodes, k, travel_mode)
    #
    # hour = 1.5
    # k = 4
    # main_route(Nodes, k, travel_mode)
    #
    # hour = 1
    # k = 3
    # main_route(Nodes, k, travel_mode)




