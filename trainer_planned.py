#Training agent: https://www.kaggle.com/tmbond/halite-example-agents
from kaggle_environments.envs.halite.helpers import *
import sys
import traceback
import copy
import math
import pprint
from random import choice, randint, shuffle

min_turns_to_spawn=100
search_depth = 4

def agent(obs, config):
    """Central function for an agent.

    Relevant properties of arguments:

    obs: 
        halite: a one-dimensional list of the amount of halite in each board space

        player: integer, player id, generally 0 or 1
            
        players: a list of players, where each is:
            [halite, { 'shipyard_uid': position }, { 'ship_uid': [position, halite] }]

        step: which turn we are on (counting up)

    Should return a dictionary where the key is the unique identifier string of a ship/shipyard
    and action is one of "CONVERT", "SPAWN", "NORTH", "SOUTH", "EAST", "WEST"
    ("SPAWN" being only applicable to shipyards and the others only to ships).
        
    """
    # Using this to avoid later computations
    genval = 1.0
            
    SHIP_MOVE_COST_RATIOS = []
    genval = 1.0
    for i in range(40):
        SHIP_MOVE_COST_RATIOS.append(genval)
        genval = genval * (1.0 - config.moveCost)
        
    reward = obs.players[obs.player][0]

    player = obs.player
    size = config.size
    board_halite = obs.halite
    board = Board(obs, config)
    player_halite, shipyards, ship_items = obs.players[player]
    shipyard_uids = list(shipyards.keys())
    shipyards = list(shipyards.values())
        
    ships = []
    ship_halite_by_pos = {}
    ship_uids = {}
    for ship_item in ship_items:
        ship_pos = ship_items[ship_item][0]
        ships.append(ship_pos)
        ship_uids[ship_pos] = ship_item
        ship_halite_by_pos[ship_pos] = ship_items[ship_item][1]
        
    action = {}
    plans = []
    updated_ships = []
        
    def get_col(pos):
        """Gets the column index of a position."""
        return pos % size

    def get_row(pos):
        """Gets the row index of a position."""
        return pos // size

    def get_col_row(pos):
        """Gets the column and row index of a position as a single tuple."""
        return (pos % size, pos // size)
        
    def manhattan_distance(pos1, pos2):
        """Gets the Manhattan distance between two positions, i.e.,
        how many moves it would take a ship to move between them."""
        # E.g. for 17-size board, 0 and 17 are actually 1 apart
        dx = manhattan_distance_single(pos1 % size, pos2 % size)
        dy = manhattan_distance_single(pos1 // size, pos2 // size)
        return dx + dy

    def manhattan_distance_single(i1, i2):
        """Gets the distance in one dimension between two columns or two rows, including wraparound."""
        iMin = min(i1, i2)
        iMax = max(i1, i2)
        return min(iMax - iMin, iMin + size - iMax)
        
    def get_new_pos(pos, direction):
        """Gets the position that is the result of moving from the given position in the given direction."""
        col, row = get_col_row(pos)
        if direction == "NORTH":
            return pos - size if pos >= size else size ** 2 - size + col
        elif direction == "SOUTH":
            return col if pos + size >= size ** 2 else pos + size
        elif direction == "EAST":
            return pos + 1 if col < size - 1 else row * size
        elif direction == "WEST":
            return pos - 1 if col > 0 else (row + 1) * size - 1

    def get_neighbors(pos):
        """Returns the possible destination positions from the given one, in the order N/S/E/W."""
        neighbors = []
        col, row = get_col_row(pos)
        neighbors.append(get_new_pos(pos, "NORTH"))
        neighbors.append(get_new_pos(pos, "SOUTH"))
        neighbors.append(get_new_pos(pos, "EAST"))
        neighbors.append(get_new_pos(pos, "WEST"))
        return neighbors

    def get_direction(from_pos, to_pos):
        """Gets the direction from one space to another, i.e., which direction a ship
        would have to move to get from from_pos to to_pos.
            
        Note this function will throw an error if used with non-adjacent spaces, so use carefully."""
        if from_pos == to_pos:
            return None
            
        neighbors = get_neighbors(from_pos)
        if to_pos == neighbors[0]:
            return "NORTH"
        elif to_pos == neighbors[1]:
            return "SOUTH"
        elif to_pos == neighbors[2]:
            return "EAST"
        elif to_pos == neighbors[3]:
            return "WEST"
        else:
            print('From:', from_pos, 'neighbors:', neighbors)
            raise Exception("Could not determine direction from " + str(from_pos) + " to " + str(to_pos))
        
    def make_plans():
        """Populates the (existing) plans array with a set of paths for all ships."""
        plans.clear()

        unplanned = copy.copy(ships)

        # Start by taking care of any dropoffs at the shipyard
        if (len(shipyards) == 1):
            shipyard = shipyards[0]
            for i in reversed(range(len(unplanned))):
                if unplanned[i] == shipyard and ship_halite_by_pos[shipyard] > 0:
                    plans.append([shipyard, shipyard])
                    unplanned.remove(shipyard)
                    break
        elif len(ships) == 1:
            # Make initial shipyard
            plans.append([ships[0], -1])
            return
            
        while len(unplanned) > 0:
            ship = unplanned.pop()

            max_halite_result = get_max_halite_per_turn([ship], search_depth, ship_halite_by_pos[ship])
            new_plan = [ship] if max_halite_result is None else max_halite_result[1]

            # Failure modes for get_max_halite_per_turn:
            # It doesn't necessarily return to the shipyard
            if new_plan[-1] != shipyard:
                new_plan = get_safe_return_path(new_plan, shipyard)

            # If it returns to the shipyard, it also need to stay there for dropoff
            if new_plan[-1] == shipyard:
                new_plan.append(shipyard)
                    
            # It can give up and just stay put, but it doesn't add a space automatically
            elif len(new_plan) == 1:
                new_plan.append(new_plan[0])
                if is_blocked(new_plan):
                    # Critical failure - tried to stay put but somebody reserved this spot.
                    # We should probably make THEM try something else instead.
                    for plan_index in range(len(plans)):
                        if len(plans[plan_index]) > 1 and plans[plan_index][1] == new_plan[1]:
                            unplanned.append(plans[plan_index][0])
                            plans.pop(plan_index)
                            break
                
            plans.append(new_plan)            
        
    def current_cell_halite(pos, starting_halite, path):
        """Gets the amount of halite left in the current cell after the given ship path is run.
            
        Does not account for the actions of other ships.
        """
        current_halite = starting_halite
        for i in range(len(path)):
            if i == 0:
                continue
            p = path[i]
            if path[i-1] == pos:
                current_halite = current_halite * 0.75
        return current_halite
        
    def is_blocked(path):
        """Checks to see if the last step in a given path is blocked by an already planned one"""
        for plan in plans:
            if len(plan) >= len(path):
                if plan[len(path) - 1] == path[-1]:
                    return True
                elif len(path) > 1 and plan[len(path) - 2] == path[-1] and plan[len(path) - 1] == path[-2]:
                    return True
        return False
        
    def get_max_halite_per_turn(path, max_depth, halite_so_far, blocked_spaces = None, debug = False):
        """Gets the most halite per turn possibly yielded by plans that go out to [max_depth] turns.
            
        Assumes if the plan does not end at the shipyard, we will then move along the shortest safe path to it.
        """
        if(len(shipyards) <= 0):
            return 0

        if max_depth == 0:
            return (halite_so_far, copy.copy(path))
            
        next_positions = get_neighbors(path[-1])
        next_positions.append(path[-1])

        choices = []

        for np in next_positions:
            path.append(np)
            if not is_blocked(path):
                new_halite = get_new_ship_halite(board_halite, path, shipyards[0], halite_so_far)
                choice = get_max_halite_per_turn(path, max_depth - 1, new_halite, blocked_spaces, debug)
                if not choice is None:
                    choices.append(choice)
            path.pop()
            
        # It is possible that we wound up in a terrible situation with no escape, including staying put
        if len(choices) == 0:
            return None
            
        best_choice = choices[0]
        best_yield = get_yield_per_turn(best_choice[1], shipyards[0], best_choice[0])

        for choice in choices[1:]:
            new_yield = get_yield_per_turn(choice[1], shipyards[0], choice[0])
            if debug:
                print(len(choice[1]), 'choice with yield', new_yield, choice[1])
            if new_yield > best_yield:
                best_choice = choice
                best_yield = new_yield

        current_yield = get_yield_per_turn(path, shipyards[0], halite_so_far)
        if current_yield > best_yield:
            if debug:
                print(len(path), 'current_yield of', current_yield, 'wins.')
            return (halite_so_far, copy.copy(path))
        else:
            if debug:
                if (best_yield > 0):
                    print(len(path), 'best yield:', best_yield,'length:', len(best_choice[1]), 'distance:', manhattan_distance(best_choice[1][-1], shipyard))
                else:
                    print(len(path), 'best yield is nothing', best_choice[1])
            return best_choice

    def get_new_ship_halite(board_halite, path, shipyard, starting_halite):
        """Determines how much halite a ship will have after following the last step in this path.
            
        Assumes that the ship has already followed all prior steps and that starting_halite is whatever
        the ship will have accumulated by then."""
        if len(path) <= 1:
            return starting_halite
        if (path[-1] == path[-2]):
            if path[-1] == shipyard and starting_halite == 0.0:
                return -100.0 # Workaround to avoid idling at shipyard
            cell_halite = current_cell_halite(path[-1], board_halite[path[-1]], path[:-1]) # Note: path[-1] because otherwise we count mining twice
            new_halite = min(starting_halite + 0.25 * cell_halite, 1000.0)
            return new_halite
        else:
            new_halite = starting_halite * (1.0 - config.moveCost)
            return new_halite

    def get_yield_per_turn(path, shipyard, halite):
        """Gets the yield, per turn, of halite following the given path."""
        if (path[-1] == shipyard):
            if len(path) == 1:
                return halite
        steps_to_dropoff = manhattan_distance(path[-1], shipyard)
        total_steps = steps_to_dropoff + len(path) - 1 # path[0] is the start, not a turn
        return halite * (SHIP_MOVE_COST_RATIOS[steps_to_dropoff]) / total_steps
        
    def get_direction(from_pos, to_pos):
        """Gets the direction from one space to another, i.e., which direction a ship
        would have to move to get from from_pos to to_pos.
            
        Note this function will throw an error if used with non-adjacent spaces, so use carefully."""
            
        if from_pos == to_pos:
            return None
            
        neighbors = get_neighbors(from_pos)
        if to_pos == neighbors[0]:
            return "NORTH"
        elif to_pos == neighbors[1]:
            return "SOUTH"
        elif to_pos == neighbors[2]:
            return "EAST"
        elif to_pos == neighbors[3]:
            return "WEST"
        else:
            print('From:', from_pos, 'neighbors:', neighbors)
            raise Exception("Could not determine direction from " + str(from_pos) + " to " + str(to_pos))
        
    def get_shortest_path(from_path, to_pos):
        """Gets the shortest paths between two spaces, or at least one of the possible shortest paths, avoiding collisions."""
            
        path = copy.copy(from_path)
            
        choices = []
            
        east = get_col(to_pos) - get_col(from_pos)
        if east < 0:
            east += size
        west = get_col(from_pos) - get_col(to_pos)
        if west < 0:
            west += size

        if west > 0 or east > 0:
            if west < east:
                for i in range(west):
                    path.append(get_new_pos(path[-1], "WEST"))
            else:
                for i in range(east):
                    path.append(get_new_pos(path[-1], "EAST"))
            
        north = get_row(from_pos) - get_row(to_pos)
        if north < 0:
            north += size
        south = get_row(to_pos) - get_row(from_pos)
        if south < 0:
            south += size

        if north < south:
            for i in range(north):
                path.append(get_new_pos(path[-1], "NORTH"))
        else:
            for i in range(south):
                path.append(get_new_pos(path[-1], "SOUTH"))

        return path
        
    def get_safe_return_path(path, to_pos, allowed_wait_steps=0):
        """Gets a return path to the spot, including waiting there (intended for shipyard dropoffs)
            
        Note this also pretty much breaks the passed-in path, so be careful when calling it.
        """
        if allowed_wait_steps > 3:
            return path
            
        result_path = get_safe_return_path_helper(copy.copy(path), to_pos, allowed_wait_steps)
            
        if not result_path is None:
            return result_path
            
        return get_safe_return_path(path, to_pos, allowed_wait_steps + 1)
            
    def get_safe_return_path_helper(path, to_pos, allowed_wait_steps=0):
        if path[-1] == to_pos and len(path) > 1 and path[-2] == to_pos:
            return path
            
        choices = []
            
        from_pos = path[-1]
            
        east = get_col(to_pos) - get_col(from_pos)
        if east < 0:
            east += size
        west = get_col(from_pos) - get_col(to_pos)
        if west < 0:
            west += size

        if west > 0 and east > 0:
            if west < east:
                choices.append(get_new_pos(from_pos, "WEST"))
            else:
                choices.append(get_new_pos(from_pos, "EAST"))
            
        north = get_row(from_pos) - get_row(to_pos)
        if north < 0:
            north += size
        south = get_row(to_pos) - get_row(from_pos)
        if south < 0:
            south += size

        if north > 0 and south > 0:
            if north < south:
                choices.append(get_new_pos(from_pos, "NORTH"))
            else:
                choices.append(get_new_pos(from_pos, "SOUTH"))
            
        for choice in choices:
            path.append(choice)
            if not is_blocked(path):
                result = get_safe_return_path_helper(path, to_pos, allowed_wait_steps)
                if not result is None:
                    return result
            path.pop()
                
        if allowed_wait_steps > 0:
            path.append(path[-1])
            if not is_blocked(path):
                result = get_safe_return_path_helper(path, to_pos, allowed_wait_steps - 1)
                if not result is None:
                    return result
            path.pop()
            
        return None
        
    try:
        make_plans()
        #print('Plans:', plans)
        for plan in plans:
            if len(plan) > 1:
                if plan[1] == -1:
                    action[ship_uids[plan[0]]] = "CONVERT"
                else:
                    direction = get_direction(plan[0], plan[1])
                    if not direction is None:
                        action[ship_uids[plan[0]]] = direction
                    updated_ships.append(plan[1])

        # Spawn Ships (whenever possible).
        if config.spawnCost <= reward and len(shipyards) == 1 and config.episodeSteps - obs.step >= min_turns_to_spawn and not shipyards[0] in updated_ships and not shipyards[0] in ships:
            reward -= config.spawnCost
            action[shipyard_uids[0]] = "SPAWN"

        return action
    except Exception as e:
        print('Error!', e)
        info = sys.exc_info()
        print('Error:', info)
        print('Traceback:', traceback.print_exception(*info))
