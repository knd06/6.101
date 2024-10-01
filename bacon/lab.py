"""
6.101 Lab:
Bacon Number
"""

#!/usr/bin/env python3

import pickle
# import typing # optional import
# import pprint # optional import

# NO ADDITIONAL IMPORTS ALLOWED!


def transform_data(raw_data):
    '''
    Mapping an actor to a list of pairs containing each actor they acted with, in the movie they acted in (except themselves)
    '''
    links = {}
    for actor1, actor2, film in raw_data:
        links.setdefault(actor1, set()).add((actor2, film))
        links.setdefault(actor2, set()).add((actor1, film))
    return links


def acted_together(transformed_data, actor_id_1, actor_id_2):
    '''
    Check if two actors have worked together
    '''
    if actor_id_1==actor_id_2:
        return True
    for id, film in transformed_data[actor_id_1]:
        if id==actor_id_2:
            return True
    return False


def actors_with_bacon_number(transformed_data, n):
    '''
    Find all of the actors who have a Bacon number n
    '''
    queue = [(4724, 0)]
    visited = set()
    res = set()
    while queue:
        actor, bacon_num = queue.pop(0)
        if bacon_num > n:
            break
        if actor not in visited:
            visited.add(actor)
            if bacon_num==n:
                res.add(actor)
            for nxt, film in transformed_data.get(actor, []):
                queue.append((nxt, bacon_num+1))
    return res


def bacon_path(transformed_data, actor_id):
    '''
    Return a "Bacon path" from Kevin Bacon to any other actor
    '''
    # queue = [[4724]]
    # visited = set()
    # while queue:
    #     path = queue.pop(0)
    #     last_actor = path[-1]
    #     for nxt, film in transformed_data.get(last_actor, []):
    #         if nxt in visited:
    #             continue
    #         else:
    #             if nxt==actor_id: 
    #                 path.append(nxt)
    #                 return path
    #             else:
    #                 visited.add(nxt)
    #                 new_path = list(path)
    #                 new_path.append(nxt)
    #                 queue.append(new_path)

    return actor_to_actor_path(transformed_data, 4724, actor_id)


def actor_to_actor_path(transformed_data, actor_id_1, actor_id_2):
    '''
    Find the minimal path between two non-Bacon actors
    '''
    if actor_id_1==actor_id_2:
        return [actor_id_1]
    queue = [[actor_id_1]]
    visited = set()
    while queue:
        path = queue.pop(0)
        last_actor = path[-1]
        for nxt, film in transformed_data.get(last_actor, []):
            if nxt in visited:
                continue
            else:
                if nxt==actor_id_2: 
                    path.append(nxt)
                    return path
                else:
                    visited.add(nxt)
                    new_path = list(path)
                    new_path.append(nxt)
                    queue.append(new_path)
    return None

def get_movie(raw_data, actor_id_1, actor_id_2):
    for dat in raw_data:
        if actor_id_1 in dat and actor_id_2 in dat:
            return dat[2]

def movie_path(transformed_data, actor_id_1, actor_id_2):
    '''
    Return the sequence of movies you could watch in order to traverse the path from one actor to another
    '''
    path = actor_to_actor_path(transformed_data, actor_id_1, actor_id_2)
    movie_path = []
    for i in range(len(path)-1):
        movie_path.append(get_movie(largedb, path[i], path[i+1]))
    movie_name = []
    for movie_id in movie_path:
        for name in moviedb.keys():
            if moviedb[name]==movie_id:
                movie_name.append(name)
    return movie_name


def actor_path(transformed_data, actor_id_1, goal_test_function):
    '''
    Return the shortest possible path from the given actor ID to any actor that satisfies the goal-test function
    '''
    if goal_test_function(actor_id_1):
        return [actor_id_1]
    paths = []
    for actor_id_2 in transformed_data:
        if goal_test_function(actor_id_2):
            path = actor_to_actor_path(transformed_data, actor_id_1, actor_id_2)
            if path is not None:
                paths.append(path)
    if paths:
        return min(paths, key=len)
    return None


def actors_connecting_films(transformed_data, film1, film2):
    '''
    Return the shortest list of actor IDs that connects the two films.
    '''
    actors_in_1 = {actor for actor, connections in transformed_data.items() if any(film == film1 for _, film in connections)}
    actors_in_2 = {actor for actor, connections in transformed_data.items() if any(film == film2 for _, film in connections)}
    queue = [[actor] for actor in actors_in_1]
    visited = set(actors_in_1)
    while queue:
        path = queue.pop(0)
        last_actor = path[-1]
        if last_actor in actors_in_2:
            return path
        for nxt, film, in transformed_data.get(last_actor, []):
            if nxt in visited:
                continue
            else:
                visited.add(nxt)
                new_path = list(path)
                new_path.append(nxt)
                queue.append(new_path)
    return None


if __name__ == "__main__":
    
    # additional code here will be run only when lab.py is invoked directly
    # (not when imported from test.py), so this is a good place to put code
    # used, for example, to generate the results for the online questions.
    
    with open("resources/names.pickle", "rb") as f:
        namedb = pickle.load(f)
    # print(type(namedb))
    # # What is Christine Forrest's ID number?
    # print(namedb['Christine Forrest']) #100570
    # # Which actor has the ID 34845? 
    # for name in namedb.keys():
    #     if namedb[name]==34845:
    #         print(name) # Lola Glaudini
    #         break
    
    with open("resources/small.pickle", "rb") as f:
        smalldb = pickle.load(f)
    # According to the small.pickle database, have Yvonne Zima and Dan Warry-Smith acted together? # Yes
    # id1 = namedb['Yvonne Zima']
    # id2 = namedb['Dan Warry-Smith']
    # print(acted_together(transform_data(smalldb), id1, id2))
    # # According to the small.pickle database, have Josef Sommer and Stig Olin acted together? # No
    # print(acted_together(transform_data(smalldb), namedb['Josef Sommer'], namedb['Stig Olin']))

    with open("resources/tiny.pickle", "rb") as f:
        tinydb = pickle.load(f)
    # print(tinydb)
    # What are the ID numbers of the actors who have a Bacon number of 0 in tiny.pickle? Enter your answer below as a Python set of integers: {4724}

    # What are the ID numbers of the actors who have a Bacon number of 1 in tiny.pickle? Enter your answer below as a Python set of integers: {2786, 1532}

    # What are the ID numbers of the actors who have a Bacon number of 2 in tiny.pickle? Enter your answer below as a Python set of integers: {1640}

    # What are the ID numbers of the actors who have a Bacon number of 3 in tiny.pickle? Enter your answer below as a Python set of integers: {} # empty

    # print(actors_with_bacon_number(transform_data(tinydb), 0))
    # print(actors_with_bacon_number(transform_data(tinydb), 1))
    # print(actors_with_bacon_number(transform_data(tinydb), 2))
    # print(actors_with_bacon_number(transform_data(tinydb), 3))

    with open("resources/large.pickle", "rb") as f:
        largedb = pickle.load(f)
    # In the large.pickle database, what is the set of actors with Bacon number 6? Enter your answer below as a Python set of strings representing actor names: ['Vjeran Tin Turk', 'Anton Radacic', 'Iva Ilakovac', 'Sven Batinic'] 
    # bacon_6 = actors_with_bacon_number(transform_data(largedb), 6)
    # bacon_6_name = []
    # for actor_id in bacon_6:
    #     for name in namedb.keys():
    #         if namedb[name]==actor_id:
    #             bacon_6_name.append(name)
    # print(bacon_6_name)

    # print(bacon_path(transform_data(largedb), 1204))
    # print(bacon_path(transform_data(largedb), 1640))
    # According to the large.pickle database, what is the path of actors from Kevin Bacon to Fylgia Zadig? Enter your answer as a Python list of actor names below: ['Kevin Bacon', 'Curtis Hanson', 'Anjelica Huston', 'Mai Zetterling', 'Anita Bjork', 'Fylgia Zadig']
    # id = namedb['Fylgia Zadig']
    # path = (bacon_path(transform_data(largedb), id))
    # path_name = []
    # for actor_id in path:
    #     for name in namedb.keys():
    #         if namedb[name]==actor_id:
    #             path_name.append(name)
    # print(path_name)

    # According to the large.pickle database, what is the minimal path of actors from Betanya Grant to J. Salome Martinez? Enter your answer as a Python list of actor names below: ['Betanya Grant', 'Jim VanBebber', 'Sage Stallone', 'Tom Gulager', 'Marc Macaulay', 'Kevin Bacon', 'J. Salome Martinez']
    # id1 = namedb['Betanya Grant']
    # id2 = namedb['J. Salome Martinez']
    # path = (actor_to_actor_path(transform_data(largedb), id1, id2))
    # path_name = []
    # for actor_id in path:
    #     for name in namedb.keys():
    #         if namedb[name]==actor_id:
    #             path_name.append(name)
    # print(path_name)

    with open("resources/movies.pickle", "rb") as f:
        moviedb = pickle.load(f)
    # print(moviedb)
    # According to the large.pickle database, what is the minimal path of movie titles connecting Richard Pierson to Vjeran Tin Turk? Enter your answer as a Python list of movie names below: ['Diner', 'Beverly Hills Cop II', '976-Evil II', 'Alien Escape', 'Ribbit', '7 Hours Later']
    # id1 = namedb['Richard Pierson']
    # id2 = namedb['Vjeran Tin Turk']
    # print(movie_path(transform_data(largedb), id1, id2))
    