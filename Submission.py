#SDK Reference: https://github.com/Kaggle/kaggle-environments/blob/39685f13f0e06ac18d4e6e09ac7c61c23db8658e/kaggle_environments/envs/halite/helpers.py#L347
# https://www.kaggle.com/superant/halite-boilerbot


# TODO:
#   Need to track individual assignments of units, if one is trying to destroy an enemy, dont let another take do the same.
#   Same is true for minign halite.  Need a unique assignment type function
#
#   Shipyards - calculate one closest to other enemy shipyards, make this spawner for attack ships
#
#   Can vary number of attack ships based on number of enemy units
#
#   Hack to make a static variable
#     agent.counter = getattr(agent, 'counter', 0) + 1
#     print(agent.counter)
#
#   Target of opportunity if returning to base
#
#   Need an objectives map so everyone doesn't try to do same thing



from kaggle_environments.envs.halite.helpers import *
import numpy as np
from math import sqrt
from enum import Enum
import random

#--------------------------------------------------------------------------------
# Global Settings
#--------------------------------------------------------------------------------

DEBUG = True

# Number of gatherers to maintain
NUMBER_OF_GATHERERS = 3
MAX_UNITS = 5

# How much more halite must be available elsewhere to bother moving off current spot
GATHER_MOVE_FACTOR = 1.2


# Proximity (Manhattan distance) away that enemies need to be for a space to be 
# considered safe
SAFE_PROXIMITY = 2

# Treshhold beyond which a friendly unit returns to base
RETURN_HALITE_THRESH = 1000

HALITE_GATHER_RATE = 0.25
HALITE_REGEN_RATE  = 1.02

SHIPYARD_HEAT_GRADE = 500
SHIP_HEAT_GRADE = 500 

initialized = False
coefficient_maps = {}

size = 0


#--------------------------------------------------------------------------------
# Global Types
#--------------------------------------------------------------------------------
class ShipTask(Enum):
    NONE = 1
    GATHER = 2
    ATTACK = 3

class CellType(Enum):
    HALITE = 1
    ENEMY_SHIP = 2
    ENEMY_SHIPYARD = 3
    FRIENDLY_SHIP = 4
    FRIENDLY_SHIPYARD = 5


# HELPER FUNCTIONS
#region 
#--------------------------------------------------------------------------------
def debug(s):
    if(DEBUG):
        print(s)

#--------------------------------------------------------------------------------
def get_direction_to(fromPos, toPos, possList=[True, True, True, True]):
    ''' Returns best direction to move from one position (fromPos) to another (toPos)
    '''
    fromX, fromY = fromPos[0], fromPos[1]
    toX, toY = toPos[0], toPos[1]
    if abs(fromX - toX) > size / 2:
        fromX += size
    if abs(fromY - toY) > size / 2:
        fromY += size
    if fromY < toY and possList[0]: return ShipAction.NORTH
    if fromY > toY and possList[1]: return ShipAction.SOUTH
    if fromX < toX and possList[2]: return ShipAction.EAST
    if fromX > toX and possList[3]: return ShipAction.WEST
    return None

#--------------------------------------------------------------------------------
def get_new_pos(pos, direction):
    '''Gets the position that is the result of moving from the given position in the given direction.
    '''
    if(direction is None):
        return pos

    new_pos = pos + direction.to_point()
    return new_pos % size

#--------------------------------------------------------------------------------
def get_neighbors(pos):
    """Returns the possible destination positions from the given one, in the order N/S/E/W."""
    neighbors = []
    for dir in ShipAction.moves():
        neighbors.append(get_new_pos(pos, dir))
    return neighbors

#--------------------------------------------------------------------------------
def manhattan_distance(pos1, pos2):
    """Gets the Manhattan distance between two positions, i.e.,
    how many moves it would take a ship to move between them. """
    dx = manhattan_distance_single(pos1.x, pos2.x)
    dy = manhattan_distance_single(pos1.y, pos2.y)
    return dx + dy

#--------------------------------------------------------------------------------
def manhattan_distance_single(i1, i2):
    """Gets the distance in one dimension between two columns or two rows, including wraparound.
    """
    iMin = min(i1, i2)
    iMax = max(i1, i2)
    return min(iMax - iMin, iMin + size - iMax)

#--------------------------------------------------------------------------------
def argmax(arr, key=None):
    return arr[arr.index(max(arr, key=key)) if key else arr.index(max(arr))]

#--------------------------------------------------------------------------------
def get_region(pos, depth = 2):
    ''' Returns a region (list of points) based on making 'depth' number of moves from a position.

        TODO: This probably isn't most efficient implementation.
    '''
    current = [pos] + get_neighbors(pos)
    destinations = current

    for i in range(2,depth):
        next_level = []
        for dest in current:
            candidates = [x for x in get_neighbors(dest) if x not in destinations]
            destinations = destinations + candidates
            next_level = next_level + candidates
        current = next_level

    return destinations

#--------------------------------------------------------------------------------
def init_coefficient_map(size):
    ''' Create a dictionary of maps which describe the distance between a position
        and all other positions on the map.  Should save computation time for
        calculating heat map
    '''
    for x in range(size): 
        for y in range(size): 
            pos = Point(x,y)
            coefficient_maps.update({pos : init_coefficients(pos, size)})

#--------------------------------------------------------------------------------
def init_coefficients(pos, size):
    ''' Given a position, creates a distance coefficient overlay for it '''
    coefficient_map = {} 
    for x in range(size): 
        for y in range(size): 
            new_pos = Point(x,y)
            coefficient_map.update({new_pos : manhattan_distance(pos,new_pos)})
    return coefficient_map

#endregion


#--------------------------------------------------------------------------------
# Agent
#--------------------------------------------------------------------------------
def agent(obs,config):
    global initialized
    global coefficient_maps
    global size

    #Grab information up front
    size = config.size
    board = Board(obs,config)
    me = board.current_player
    opponents = board.opponents

    #INITIALIZATION CODE
    #region INIT
    if not initialized:
        print("Initializing")
        init_coefficient_map(size)
        initialized = True
    #endregion

    #TEST CODE
    agent.counter = getattr(agent, 'counter', 0) + 1
    print("Turn " + str(agent.counter))


    # CONTROL HELPERS
    #region HELPER FUNCTIONS

    #--------------------------------------------------------------------------------
    def get_safe_moves(pos, halite_level):
        ''' Loop through possible moves and determine which are likely to be safe 
            based on proximity of enemies
        '''
        move_list = ShipAction.moves() 
        good_moves = []

        region = get_region(pos, SAFE_PROXIMITY)
        if check_region_for_enemy(region, halite_level):
            good_moves.append(None)
                
        for move in move_list:
            new_pos = get_new_pos(pos, move)
            region = get_region(new_pos, SAFE_PROXIMITY)
            if check_region_for_enemy(region, halite_level, task = 'AVOID'):
                good_moves.append(move)

    #--------------------------------------------------------------------------------
    def check_region_for_enemy(region, halite_level, task = 'AVOID'):
        ''' Check a region to see if there is an enemy within it.
            Handles avoid (look for ships with less halite) and attack
            (look for ships with more halite)
        '''
        found = False
        for pos in region:
            cell = board.cells[pos]

            if (cell.ship is not None and 
                cell.ship.player_id != me):
                
                if(task == 'ATTACK'):
                    if(cell.ship.halite > halite_level):
                        found = True
                        next_map.add_target(cell.ship.id, pos)
                        break

                if(task == 'AVOID'):
                    if(cell.ship.halite < halite_level):
                        found = True
                        next_map.add_agressor(cell.ship.id, pos)
                        break

                else:
                    print("check_region_for_enemy task error")
                    
        return found



    #--------------------------------------------------------------------------------
    def find_halite(ship, depth):
        ''' Find the maximum halite available assuming we want to make 'Depth' number
            of moves.  
        '''
        destinations = get_region(ship.position, depth)
        best = argmax([item.to_index(size) for item in destinations], key=obs.halite.__getitem__)
        return Point.from_index(best, size)

    #--------------------------------------------------------------------------------
    def halite_prediction(position, turns):
        ''' Return how much halite would be gathered by sitting on position for X turns.
            Useful for determining if we should move or stay put.
        '''
        current = board.cells[position].halite
        total = 0

        for i in range(turns):
            gathered = HALITE_GATHER_RATE * current
            total += gathered
            current = HALITE_REGEN_RATE * (current - gathered)

        return total
       
    #--------------------------------------------------------------------------------
    def closest_shipyard(ship):
        ''' Find closest shipyard and return to it.

            Returns None if no shipyards exist, need to check for this and take appropriate
            action.

            #TODO: Work on what to do if no shipyards
        '''
        if(len(me.shipyards) == 0):
            return ShipAction.CONVERT

        shortest = 1000000
        target = None
        for shipyard in me.shipyards:
            distance = manhattan_distance(ship.position, shipyard.position)
            if(distance < shortest):
                target = shipyard
                shortest = distance

        return target

    #endregion


    # GAME MEMORY
    #region GAME MEMORY

    #--------------------------------------------------------------------------------
    class heat(object):
        ''' Cells for my map.  Hopefully this doesn't get confusing with the other cells
        '''
        #---------------------------------------------------------------------------
        def __init__(self, position):
            self._position = position
            self.score = self._score()
            self.type = None
            self.target_count = 0

        #---------------------------------------------------------------------------
        def occupied(self, is_occupied = True):
            if(is_occupied):
                self.score = -1
            else:
                self.score = _score()

        #---------------------------------------------------------------------------
        def _score(self):
            shipyard = board.cells[self._position].shipyard
            ship = board.cells[self._position].ship

            if(shipyard != None and shipyard.player_id != me):
                self.type = CellType.ENEMY_SHIPYARD
                return SHIPYARD_HEAT_GRADE

            elif(ship != None and ship.player_id != me):
                self.type = CellType.ENEMY_SHIP
                return SHIP_HEAT_GRADE + ship.halite

            else:
                self.type = CellType.HALITE
                return board.cells[self._position].halite

        #---------------------------------------------------------------------------
        def __str__(self) -> str:
            return str(self.score)

    #--------------------------------------------------------------------------------
    class map(object):
        ''' Map class similar to the board object.  Board object includes a .next() function
            that will propogate actions in the future, but it seemed easier just to plop objects
            where I want them instead of iteratively calling next as I planned.  I also may want
            to go multiple turns in future.  
        '''
        #---------------------------------------------------------------------------
        def __init__(self, size):
            self.cells: Dict[Point, heat] = {}
            for x in range(size): 
                for y in range(size): 
                    pos = Point(x,y)
                    self.cells[pos] = heat(pos)
            self.size = size
           
        #--------------------------------------------------------------------------------
        def add_occupier(self, unit_id, position):
            self.cells[position].occupied()

        #--------------------------------------------------------------------------------
        def is_occupied(self, position):
            if(self.cells[position].score == -1):
                return True
            else:
                return False

        #---------------------------------------------------------------------------
        def __mul__(self, other):
            result = {}
            for x in range(size): 
                for y in range(size): 
                    pos = Point(x,y)
                    result.update({pos : self.cells[pos].score * other[pos]})
            return result

        #--------------------------------------------------------------------------------
        def __str__(self) -> str:
            '''
            Use same string method as the board.
            '''
            just_len = 5
            size = self.size
            result = ''
            for y in range(size):
                result += str(y).rjust(3)
                for x in range(size):
                    val = str(self.cells[(x, size - y - 1)])

                    result += '|'
                    result += val.rjust(just_len) if val is not None else ' '.rjust(just_len)
                    
                result += '|\n'

            result += ' '.rjust(just_len) 
            for x in range(size):
                result += str(x).center(just_len + 1) 
            result += '\n'
            return result

    #endregion

    #region CONTROL FUNCTIONS

    #--------------------------------------------------------------------------------
    def position_deconflict(ship, dir):
        ''' Function to handle deconfliction when a ship wants to move into an occupied
            space.  Right now, going to iterate through directions.  Later may come up
            with something better.  
        ''' 
        debug("position_deconflict: checking " + str(dir) + " from " + ship.id)

        if(dir == ShipAction.NORTH):
            choices = [ShipAction.EAST, ShipAction.WEST, None, ShipAction.SOUTH]
        elif(dir == ShipAction.SOUTH):
            choices = [ShipAction.EAST, ShipAction.WEST, None, ShipAction.NORTH]
        elif(dir == ShipAction.EAST):
            choices = [ShipAction.NORTH, ShipAction.SOUTH, None, ShipAction.WEST]
        elif(dir == ShipAction.WEST):
            choices = [ShipAction.NORTH, ShipAction.SOUTH, None, ShipAction.EAST]
        else:
            choices = [ShipAction.NORTH, ShipAction.SOUTH, ShipAction.EAST, ShipAction.WEST]

        choices = [dir] + choices;
        for choice in choices:
            new_pos = get_new_pos(ship.position, choice)
            if not next_map.is_occupied(new_pos):
                next_map.add_occupier(ship.id, new_pos)
                debug("position_deconflict : wanted " + str(dir) + " ended up with " + str(choice))
                return choice;

        print("position_deconflict error")
        return None

    #--------------------------------------------------------------------------------
    def ship_control(ship):
        ''' Function to handle the details of assigning moves to specific ships.
            TODO: Better logic for spawning shipyards
        '''
        print("Controlling ship " + ship.id)

        # First see if need to make a shipyard
        # TODO: More advanced later
        if len(me.shipyards) == 0:
            print("ship_control: Converting to shipyard")
            ship.next_action = ShipAction.CONVERT
            return

        dest = ship.position
        # Return to base if collected a lot of halite
        if ship.halite > RETURN_HALITE_THRESH:
            shipyard = closest_shipyard(ship)
            dest = shipyard.position

        else:
            print("\n\nMoving ship")
            start_map = heat_map 
            print("\nStart map")
            print(start_map)
            coef_map = coefficient_maps[ship.position]
            print("\nCoef map")
            print(coef_map)
            the_map = heat_map * coef_map
            print("\nCombined map")
            print(the_map)

            pass

        dir = get_direction_to(ship.position, dest)

        dir = position_deconflict(ship, dir)              # Make sure we don't run into our own units

        if(dir is not None):
            ship.next_action = dir
            print("ship_gather: next dir " + str(dir))
  
    #--------------------------------------------------------------------------------
    def shipyard_control(shipyard):

        # TODO: Needs updating for multiple shipyards

        # TODO: Need better logic for ship sitting on shipyard

        # TODO: Spawn a ship if enemy close

        # If there are no ships, use first shipyard to spawn a ship.
        if len(me.ships) < MAX_UNITS:
            if not heat_map.is_occupied(shipyard.position):
                shipyard.next_action = ShipyardAction.SPAWN
            else:
                print("Shipyard blocked from creating by ship sitting on it.")
           
    #endregion


    #--------------------------------------------------------------------------------
    # Actual Function Code
    #--------------------------------------------------------------------------------
    heat_map = map(size)
    print(board)
 
    # Set actions for each ship
    for ship in me.ships:
        ship_control(ship)

    print(heat_map)

    # Set actions for each shipyard
    for shipyard in me.shipyards:
        shipyard_control(shipyard)
    
    return me.next_actions