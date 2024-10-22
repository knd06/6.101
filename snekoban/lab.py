"""
6.1010 Lab:
Snekoban Game
"""

import json # optional import for loading test_levels
import typing # optional import
import pprint # optional import

# NO ADDITIONAL IMPORTS!


DIRECTION_VECTOR = {
    "up": (-1, 0),
    "down": (+1, 0),
    "left": (0, -1),
    "right": (0, +1),
}


def make_new_game(level_description):
    """
    Given a description of a game state, create and return a game
    representation of your choice.

    The given description is a list of lists of lists of strs, representing the
    locations of the objects on the board (as described in the lab writeup).

    For example, a valid level_description is:

    [
        [[], ['wall'], ['computer']],
        [['target', 'player'], ['computer'], ['target']],
    ]

    The exact choice of representation is up to you; but note that what you
    return will be used as input to the other functions.
    """
    level = {
        'height': len(level_description),
        'width': len(level_description[0])
    }
    # dict connecting object types to sets of location tuples
    for r in range(len(level_description)):
        for c, e in enumerate(level_description[r]):
            for object in e:
                level.setdefault(object, set()).add((r, c))
    return level


def victory_check(game):
    """
    Given a game representation (of the form returned from make_new_game),
    return a Boolean: True if the given game satisfies the victory condition,
    and False otherwise.
    """
    if len(game.get('computer', set())) == 0 or len(game.get('target', set())) == 0:
        return False
    return game['computer'] == game['target']


def step_game(game, direction):
    """
    Given a game representation (of the form returned from make_new_game),
    return a game representation (of that same form), representing the
    updated game after running one step of the game.  The user's input is given
    by direction, which is one of the following:
        {'up', 'down', 'left', 'right'}.

    This function should not mutate its input.
    """
    move = DIRECTION_VECTOR[direction]
    new_game = {key: value.copy() if isinstance(value, set) else value for key, value in game.items()}
    player = next(iter(new_game['player']))
    new_player = (player[0] + move[0], player[1] + move[1])

    # move check
    if not (0 <= new_player[0] < new_game['height'] and 0 <= new_player[1] < new_game['width']):
        return game  # out of bounds
    if new_player in new_game.get('wall', set()):
        return game  # move into wall

    if new_player in new_game.get('computer', set()):
        new_computer = (new_player[0] + move[0], new_player[1] + move[1]) # new computer position
        if (0 <= new_computer[0] < new_game['height'] and 0 <= new_computer[1] < new_game['width']
                and new_computer not in new_game.get('wall', set())
                and new_computer not in new_game.get('computer', set())):
            new_game['computer'].remove(new_player)
            new_game['computer'].add(new_computer)
        else:
            return game  # computer blocked

    new_game['player'].remove(player)
    new_game['player'].add(new_player)

    return new_game


def dump_game(game):
    """
    Given a game representation (of the form returned from make_new_game),
    convert it back into a level description that would be a suitable input to
    make_new_game (a list of lists of lists of strings).

    This function is used by the GUI and the tests to see what your game
    implementation has done, and it can also serve as a rudimentary way to
    print out the current state of your game for testing and debugging on your
    own.
    """
    level_description = [[[] for _ in range(game['width'])] for _ in range(game['height'])]
    for obj, positions in game.items():
        if isinstance(positions, set):
            for r, c in positions:
                level_description[r][c].append(obj)
    return level_description


def solve_puzzle(game):
    """
    Given a game representation (of the form returned from make_new_game), find
    a solution.

    Return a list of strings representing the shortest sequence of moves ("up",
    "down", "left", and "right") needed to reach the victory condition.

    If the given level cannot be solved, return None.
    """
    def game_state(game):
        '''
        Create a hashable key for the game state to check for repeats.
        '''
        return (frozenset(game['player']), frozenset(game['computer']))

    queue = [(game, [])]
    visited = {game_state(game)}
    while queue:
        current_game, path = queue.pop(0)
        if victory_check(current_game):
            return path
        for dir in DIRECTION_VECTOR:
            nxt_game = step_game(current_game, dir)
            nxt_key = game_state(nxt_game)

            if nxt_key not in visited:
                visited.add(nxt_key)
                queue.append((nxt_game, path + [dir]))

    return None # No solution found


if __name__ == "__main__":
    pass
    with open('puzzles/m1_008.json', 'r') as f:
        level = json.load(f)

    game = make_new_game(level_description=level)
    solution = solve_puzzle(game)
    if solution:
        print(solution)
    else:
        print('No solution found')