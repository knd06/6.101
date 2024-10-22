"""
6.101 Lab:
Recipes
"""

import pickle
import sys
# import typing # optional import
# import pprint # optional import

sys.setrecursionlimit(20_000)
# NO ADDITIONAL IMPORTS!


def atomic_ingredient_costs(recipes_db):
    """
    Given a recipes database, a list containing compound and atomic food tuples,
    make and return a dictionary mapping each atomic food name to its cost.
    """
    atomic_costs = {}
    for item in recipes_db:
        if item[0] == 'atomic':
            atomic_costs[item[1]] = item[2]
    return atomic_costs


def compound_ingredient_possibilities(recipes_db):
    """
    Given a recipes database, a list containing compound and atomic food tuples,
    make and return a dictionary that maps each compound food name to a
    list of all the ingredient lists associated with that name.
    """
    compound_possibilities = {}
    for item in recipes_db:
        if item[0] == 'compound':
            if item[1] not in compound_possibilities:
                compound_possibilities[item[1]] = []
            compound_possibilities[item[1]].append(item[2])
    return compound_possibilities


def lowest_cost(recipes_db, food_name, forbidden=None):
    """
    Given a recipes database and the name of a food (str), return the lowest
    cost of a full recipe for the given food item or None if there is no way
    to make the food_item.
    """
    raise NotImplementedError    


def scaled_recipe(recipe_dict, n):
    """
    Given a dictionary of ingredients mapped to quantities needed, returns a
    new dictionary with the quantities scaled by n.
    """
    raise NotImplementedError


def add_recipes(recipe_dicts):
    """
    Given a list of recipe dictionaries that map food items to quantities,
    return a new dictionary that maps each ingredient name
    to the sum of its quantities across the given recipe dictionaries.

    For example,
        add_recipes([{'milk':1, 'chocolate':1}, {'sugar':1, 'milk':2}])
    should return:
        {'milk':3, 'chocolate': 1, 'sugar': 1}
    """
    raise NotImplementedError


def cheapest_flat_recipe(recipes_db, food_name):
    """
    Given a recipes database and the name of a food (str), return a dictionary
    (mapping atomic food items to quantities) representing the cheapest full
    recipe for the given food item.

    Returns None if there is no possible recipe.
    """
    raise NotImplementedError


def combine_recipes(nested_recipes):
    """
    Given a list of lists of recipe dictionaries, where each inner list
    represents all the recipes for a certain ingredient, compute and return a
    list of recipe dictionaries that represent all the possible combinations of
    ingredient recipes.
    """
    raise NotImplementedError


def all_flat_recipes(recipes_db, food_name):
    """
    Given a recipes database, the name of a food (str), produce a list (in any
    order) of all possible flat recipe dictionaries for that category.

    Returns an empty list if there are no possible recipes
    """
    raise NotImplementedError


if __name__ == "__main__":
    # load example recipes from section 3 of the write-up
    with open("test_recipes/example_recipes.pickle", "rb") as f:
        example_recipes_db = pickle.load(f)
    # you are free to add additional testing code here!

    # atomic_cnt = 0
    # for item in example_recipes_db:
    #     if item[0] == 'atomic':
    #         atomic_cnt += 1
    # print(atomic_cnt)

    # compound_set = set()
    # for item in example_recipes_db:
    #     if item[0] == 'compound':
    #         compound_set.add(item[1])
    # print(len(compound_set))

    # atomic_costs = atomic_ingredient_costs(example_recipes_db)
    # total_cost = sum(atomic_costs.values())
    # print(total_cost)

    compound_possibilities = compound_ingredient_possibilities(example_recipes_db)
    ways_cnt = sum(1 for item in compound_possibilities if len(compound_possibilities[item]) > 1)
    print(ways_cnt)

