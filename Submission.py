#SDK Reference: https://github.com/Kaggle/kaggle-environments/blob/39685f13f0e06ac18d4e6e09ac7c61c23db8658e/kaggle_environments/envs/halite/helpers.py#L347
# https://www.kaggle.com/superant/halite-boilerbot


# TODO:
#   Tech Debt: There is a lot of using global variable holders I don't like, but stops from excessively passing variables for now
#
#   Shipyards - calculate one closest to other enemy shipyards, make this spawner for attack ships
#
#   Weight for returning to base, based on halite we have.
#

from kaggle_environments.envs.halite.helpers import *
import numpy as np
from math import sqrt
from enum import Enum
import random

#--------------------------------------------------------------------------------
# Tunable Parameters
#--------------------------------------------------------------------------------
DEBUG = True

SHIPYARD_HEAT_GRADE = 0
SHIP_HEAT_GRADE = 200 

UNITS_PER_ARMADA = 5

MAX_ARMADAS = 6

# Give preference to not moving (i.e. if on a decent halite deposit, don't
# move to slightly better).  Intent is for this to increase value of a square
# a unit is currently on, therefore giving it prefernece for staying there.
# TODO: Not implemented yet
STAY_PUT_BONUS = 1.25

# If true, will use the distance to map starting point (makes assumption there
# is a shipyard there) in opportunity cost map creation
# TODO: Not implemented yet
USE_ROUND_TRIP = True

# Number of gatherers to maintain
NUMBER_OF_GATHERERS = 3

# Proximity (Manhattan distance) away that enemies need to be for a space to be 
# considered safe
SAFE_PROXIMITY = 2

# Treshhold beyond which a friendly unit returns to base, this is multiplicative
# with the average halite on the board (i.e. if board is heavily mined we don't
# want a huge threshold so we keep the stream of halite going)
RETURN_HALITE_THRESH = 12


#--------------------------------------------------------------------------------
# Global Balues
#--------------------------------------------------------------------------------

# Describes rules of the game for Haltie generation
HALITE_GATHER_RATE = 0.25
HALITE_REGEN_RATE  = 1.02

initialized = False
coefficient_maps = {}
opportunity_cost = {}

size = 0
board = 0
starting_pos = 0


#--------------------------------------------------------------------------------
# Global Types
#--------------------------------------------------------------------------------
class ArmadaTask(Enum):
    GATHER_CLOSE = 1
    GATHER_MEDIUM = 2
    GATHER_FAR = 3
    ATTACK = 4
    OPPURTUNITY = 5

armada_count = 0
armada_tasks = [ArmadaTask.GATHER_CLOSE, ArmadaTask.GATHER_CLOSE, ArmadaTask.GATHER_MEDIUM, ArmadaTask.GATHER_FAR, ArmadaTask.ATTACK, ArmadaTask.OPPURTUNITY]

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
def dictmax(dictionary):  
     v=list(dictionary.values())
     k=list(dictionary.keys())
     max_v = max(v)
     return k[v.index(max_v)], max_v

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
def get_avg_halite():
    total = 0
    cnt = 0
    for key, values in board.cells.items():
        total += values.halite
        cnt += 1
    return total/cnt

#--------------------------------------------------------------------------------
def init_coefficient_map(size):
    ''' Create a dictionary of maps which describe the distance between a position
        and all other positions on the map.  Should save computation time for
        calculating heat map
    '''
    calc_opportunity_array()

    for x in range(size): 
        for y in range(size): 
            pos = Point(x,y)
            coefficient_maps.update({pos : init_coefficients(pos, size)})

#--------------------------------------------------------------------------------
def init_coefficients(pos, size):
    ''' Given a position, create the opportunity cost coefficient for all other map
        positions.  Opportunity cost is a function of distance and the halite gather rate
    '''
    coefficient_map = {} 
    for x in range(size): 
        for y in range(size): 
            new_pos = Point(x,y)
            coefficient_map.update({new_pos : opportunity_cost[manhattan_distance(pos,new_pos)]})
    return coefficient_map

#--------------------------------------------------------------------------------
def calc_opportunity_array(max_length = 40):
    ''' Calculates coefficients on for how much halite would be gathered at current
        position for various turn lengths (in essence compound depreciation).  
    '''
    temp = {}
    current = 1.0
    total = 0
    opportunity_cost.update({0 : total})
    for i in range(1,max_length):
        #gathered = HALITE_GATHER_RATE * current
        #total += gathered
        #current = HALITE_REGEN_RATE * (current - gathered)
        total += 2*HALITE_GATHER_RATE
        opportunity_cost.update({i : total})


#--------------------------------------------------------------------------------
def map_to_string(map):
    ''' Print a point based dictionary in a readable, 2D fashion
    '''
    just_len = 5
    result = ''
    for y in range(size):
        result += str(20-y).rjust(3)
        for x in range(size):
            val = '%.5s' % str(map[(x, size - y - 1)])
            result += '|'
            result += val.rjust(just_len) if val is not None else ' '.rjust(just_len)
                    
        result += '|\n'

    result += ' '.rjust(just_len) 
    for x in range(size):
        result += str(x).center(just_len + 1) 
    result += '\n'
    return result
#endregion


#--------------------------------------------------------------------------------
# Agent
#--------------------------------------------------------------------------------
def agent(obs,config): 

    #Grab information up front
    global size
    size = config.size

    global board
    board = Board(obs,config)

    me = board.current_player
    opponents = board.opponents

    avg_halite = get_avg_halite()
    print("Average halite: " + str(avg_halite))

    #INITIALIZATION CODE
    #region INIT
    global initialized
    global coefficient_maps
    global opportunity_cost
    global starting_pos
    global armada_count
    if not initialized:
        print("Initializing")
        init_coefficient_map(size)
        initialized = True
        starting_pos = me.ships[0].position
        armada_count = 0
        
    #endregion

    #TEST CODE
    agent.counter = getattr(agent, 'counter', 0) + 1
    print("Turn " + str(agent.counter))


    # CONTROL HELPERS
    #region HELPER FUNCTIONS

    #--------------------------------------------------------------------------------
    def get_safe_moves(pos, my_ship_halite):
        ''' Loop through possible moves and determine which are likely to be safe 
            based on proximity of enemies
        '''
        move_list = ShipAction.moves() 
        good_moves = []

        region = get_region(pos, SAFE_PROXIMITY)
        if check_region_for_enemy(region, my_ship_halite):
            good_moves.append(None)
                
        for move in move_list:
            new_pos = get_new_pos(pos, move)
            region = get_region(new_pos, SAFE_PROXIMITY)
            if check_region_for_enemy(region, my_ship_halite, task = 'AVOID'):
                good_moves.append(move)

    #--------------------------------------------------------------------------------
    def check_region_for_enemy(region, task = 'AVOID', my_ship_halite = 0):
        ''' Check a region to see if there is an enemy within it.
            Handles avoid (look for ships with less halite) and attack
            (look for ships with more halite)
        '''
        found = False
        for pos in region:
            cell = board.cells[pos]

            if (cell.ship is not None and 
                cell.ship.player_id != me.id):
                
                if(task == 'ATTACK'):
                    if(cell.ship.halite > my_ship_halite):
                        found = True
                        #next_map.add_target(cell.ship.id, pos)
                        break

                elif(task == 'AVOID'):
                    if(cell.ship.halite < my_ship_halite):
                        found = True
                        #next_map.add_agressor(cell.ship.id, pos)
                        break

                elif(task == 'SHIPYARD_DEFENSE'):
                    print('check_region_for_enemy: Enemy near shipyard: unit ' + str(cell.ship.id))
                    found = True
                    break

                else:
                    print("check_region_for_enemy task error")
                    
        return found
      
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


# HEAT MAP
#region HEAT MAP
    #--------------------------------------------------------------------------------
    class HeatCell(object):
        ''' Cells for my map.  Hopefully this doesn't get confusing with the other cells
        '''
        #---------------------------------------------------------------------------
        def __init__(self, position):
            self._position = position
            self.score = self._score()
            self.type = None
            self.target_count = 0
            self._occupied = False

        #---------------------------------------------------------------------------
        def occupied(self, is_occupied = True):
            self._occupied = is_occupied

        def is_occupied(self):
            return self._occupied

        #---------------------------------------------------------------------------
        def _score(self):
            shipyard = board.cells[self._position].shipyard
            ship = board.cells[self._position].ship

            if(shipyard != None and shipyard.player_id != me.id):
                self.type = CellType.ENEMY_SHIPYARD
                return SHIPYARD_HEAT_GRADE

            elif(ship != None and ship.player_id != me.id):
                self.type = CellType.ENEMY_SHIP
                return SHIP_HEAT_GRADE# + ship.halite)/avg_halite

            else:
                self.type = CellType.HALITE
                return board.cells[self._position].halite

        #---------------------------------------------------------------------------
        def __str__(self) -> str:
            return str(self.score)

    #--------------------------------------------------------------------------------
    class Map(object):
        ''' Map class similar to the board object.  Board object includes a .next() function
            that will propogate actions in the future, but it seemed easier just to plop objects
            where I want them instead of iteratively calling next as I planned.  I also may want
            to go multiple turns in future.  
        '''
        #---------------------------------------------------------------------------
        def __init__(self, size):
            self.cells: Dict[Point, HeatCell] = {}
            for x in range(size): 
                for y in range(size): 
                    pos = Point(x,y)
                    self.cells[pos] = HeatCell(pos)
            self.size = size

        #---------------------------------------------------------------------------
        def max(self):
            return dictmax(self.cells)
           
        #---------------------------------------------------------------------------
        def add_occupier(self, unit_id, position):
            self.cells[position].occupied()

        #---------------------------------------------------------------------------
        def is_occupied(self, position):
            return self.cells[position].is_occupied()

        #---------------------------------------------------------------------------
        def create_opportunity_map(self, base_case, coeff_map):
            ''' Calculating opportunity cost by subtracting out how much halite we
                could have gathered if we stayed put rather than moved to a new location.
            '''
            result = {}
            for x in range(size): 
                for y in range(size): 
                    pos = Point(x,y)
                    cost = self.cells[pos].score - base_case * coeff_map[pos]
                    result.update({pos : cost})
            return result

        #---------------------------------------------------------------------------
        def calc_opportunity_cost(self, cell_score, base_case,  coeff):
            return 

        #--------------------------------------------------------------------------------
        def __str__(self) -> str:
            return map_to_string(self.cells)
#end region

# SHIP/ARMADA CLASSES
#region 
    #--------------------------------------------------------------------------------
    class ShipMeta(object):
        def __init__(self, ship):
            self.id = ship.id
            self.position = ship.position
            self.ship = ship
            self.heat_map = None
            self.destination = None
            self.target_unit = None

        #---------------------------------------------------------------------------
        def create_heat_map(self, base_map, coeff_map):
            self.heat_map = base_map.create_opportunity_map(avg_halite, coeff_map)

            # Scrubs the heat map to make sure this ship doesn't pursue enemy ships with more halite 
            # TODO: Look for cleaner implementation 
            for player in opponents: #TODO: Global variable dependency
                for enemy in player.ships:
                    if(enemy.halite <= self.ship.halite):
                        self.heat_map[enemy.position] = -1000
            return

        #---------------------------------------------------------------------------
        def max(self):
            pos, val = dictmax(self.heat_map)
            self.destination = pos
            return val

        #--------------------------------------------------------------------------------
        def __str__(self) -> str:
            result = 'Meta:' + self.id + " at "+ str(self.position) + " dest " + str(self.destination)
            return result

    #--------------------------------------------------------------------------------
    class Armada(object):
        def __init__(self, task):
            self.ships = []
            self.task = task

        def add_ship(self, ship):
            self.ships.append(ship)

        def remove_ship(self, ship_id):
            pass

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
            if not heat_map.is_occupied(new_pos):
                heat_map.add_occupier(ship.id, new_pos)
                debug("position_deconflict: " + ship.id + ":" + str(ship.position) + " will move to: " + str(new_pos))
                debug("position_deconflict : wanted " + str(dir) + " ended up with " + str(choice))
                return choice;
            else:
                debug("position_deconflict: " + ship.id + ":" + str(ship.position) + " wanted " + str(new_pos) + " but it was occupied.")

        print("position_deconflict error")
        return None

    #--------------------------------------------------------------------------------
    def shipyard_control(shipyard):
        # TODO: Needs updating for multiple shipyards

        # TODO: Need better logic for ship sitting on shipyard

        # TODO: Spawn a ship if enemy close

        spawn = False

        if(me.halite < 500):
            spawn = False

        if(heat_map.is_occupied(shipyard.position)):
            spawn = False
            print("Shipyard " + shipyard.id + " blocked from creating by ship sitting on it.")

        elif len(me.ships) < MAX_UNITS:
            spawn = True

        elif check_region_for_enemy(get_neighbors(shipyard.position), 'SHIPYARD_DEFENSE'):
            print("Shipyard " + shipyard.id + " spawning defensive ship.")
            spawn = True

        if spawn:
            shipyard.next_action = ShipyardAction.SPAWN

        return

    #--------------------------------------------------------------------------------
    def ship_preprocess(meta):
        ''' Handles pre-processing steps where we want hardcoded actions for ships
            in specific situations.
        '''
        print("Controlling ship " + meta.id + " halite: " + str(meta.ship.halite))

        # First see if need to make a shipyard
        if len(me.shipyards) == 0:
            debug("ship_control: Converting to shipyard")
            meta.ship.next_action = ShipAction.CONVERT



        # Return to base if collected a lot of halite
        elif ship.halite > RETURN_HALITE_THRESH * avg_halite:
            shipyard = closest_shipyard(ship)
            meta.destination = shipyard.position
            print(ship.id + " is full.  Returning to " + str(meta.destination))

        else:
            #print(heat_map)

            #print(map_to_string(coefficient_maps[ship.position]))

            meta.create_heat_map(heat_map, coefficient_maps[ship.position])   

            #print(map_to_string(meta.heat_map))

            to_rank.append(meta)
            #print("Added to rank matrix:")
            #for m in to_rank:
            #    print(m)

        meta_ships.append(meta)
        #print("Added to Meta_Ships:")
        #for m in meta_ships:
        #    print(m)
        
    #--------------------------------------------------------------------------------
    def ship_execute(meta):
        ''' Final logic for translating meta actions into actual actions
        '''
        print("ship_execute for " + meta.id)
        if meta.ship.next_action is None:
            dir = get_direction_to(meta.ship.position, meta.destination)
            dir = position_deconflict(meta.ship, dir)              # Make sure we don't run into our own units
            if(dir is not None):
                meta.ship.next_action = dir
                print("ship_execute: next dir " + str(dir))

        return

    #--------------------------------------------------------------------------------
    def process_rank(rank_list, remove_point = None):
        if not rank_list:
            return

        max_val = None
        max_meta = None

        debug("\n\nprocess_rank")

        for meta in rank_list:
            debug(meta)

            if remove_point is not None:
                meta.heat_map[remove_point] = float("-inf")
                
            val = meta.max()
            debug("Local max: " + str(val) + " at " + str(meta.destination))
            if(max_val is None or val > max_val):
                debug("This was max val.")
                max_val = val
                max_meta = meta

        debug("Selected: " + max_meta.id + " to go to: +" + str(meta.destination))
        rank_list.remove(max_meta)
        process_rank(rank_list, max_meta.destination)   
              
    #endregion


    #--------------------------------------------------------------------------------
    # Actual Function Code
    #--------------------------------------------------------------------------------
    heat_map = Map(size)
    print(board)
    
    print(heat_map)
     
    # Set actions for each ship
    to_rank = []
    meta_ships = []

    for ship in me.ships:
        meta = ShipMeta(ship)
        ship_preprocess(meta)
        
    process_rank(to_rank)

    for m in meta_ships:
        ship_execute(m)

   
    # Set actions for each shipyard
    for shipyard in me.shipyards:
        shipyard_control(shipyard)
    
    return me.next_actions