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
import math
from enum import Enum
import random
from operator import attrgetter

#--------------------------------------------------------------------------------
# Tunable Parameters
#--------------------------------------------------------------------------------
DEBUG = False

MAX_UNITS = 20

#Heat grade for an enemy shipyard, setting this high will make our units want to attack enemy shipyards
SHIPYARD_HEAT_GRADE = 0

#Heat grade for an enemy ships, SHIP_USE_CARGO makes the halite onboard an enemy ship addative (will target enemies with lots of halite)
SHIP_HEAT_GRADE = 0 
SHIP_USE_CARGO = False    

# Give preference to not moving (i.e. if on a decent halite deposit, don't
# move to slightly better).  Intent is for this to increase value of a square
# a unit is currently on, therefore giving it prefernece for staying there.
STAY_PUT_BONUS = 2.0

CLOSE_TO_HOME_MULTI = 1.5

# Proximity (Manhattan distance) away that enemies need to be for a space to be 
# considered safe
SAFE_PROXIMITY = 2

# Treshhold beyond which a friendly unit returns to base, this is multiplicative
# with the average halite on the board (i.e. if board is heavily mined we don't
# want a huge threshold so we keep the stream of halite going)
RETURN_HALITE_THRESH = 20
RETURN_HALITE_MIN = 1000

#--------------------------------------------------------------------------------
# Global Values
#--------------------------------------------------------------------------------
# Describes rules of the game for Haltie generation
HALITE_GATHER_RATE = 0.25
HALITE_REGEN_RATE  = 1.00

initialized = False
coefficient_maps = {}
shipyard_maps = {}
opportunity_cost = {}

size = 0
board = 0
starting_pos = 0
me = 0
opponents = 0

#--------------------------------------------------------------------------------
# Global Types
#--------------------------------------------------------------------------------

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
    def create_heat_map(self, base_map, base_case, coeff_map, return_map):

        final_coeff_map = map_add(return_map, coeff_map)

        self.heat_map = base_map.create_opportunity_map(base_case, final_coeff_map)

        # Favor staying on current position if it has decent amount of halite
        self.heat_map[self.position] = STAY_PUT_BONUS * self.heat_map[self.position] 

        # Scrubs the heat map to make sure this ship doesn't pursue enemy ships with more halite 
        # TODO: Look for cleaner implementation 
        for player in opponents: #TODO: Global variable dependency
            for enemy in player.ships:
                if(enemy.halite <= self.ship.halite):
                    self.heat_map[enemy.position] = float("-inf")      
        return

    #---------------------------------------------------------------------------
    def max(self):
        pos, val = map_max(self.heat_map)
        self.destination = pos
        return val

    #--------------------------------------------------------------------------------
    def __str__(self) -> str:
        result = 'Meta:' + self.id + " at "+ str(self.position) + " dest " + str(self.destination)
        return result

# HEAT MAP
#region
#--------------------------------------------------------------------------------
class HeatCell(object):
    ''' Cells for my map.  Hopefully this doesn't get confusing with the other cells
    '''
    #---------------------------------------------------------------------------
    def __init__(self, position):
        self._position = position
        self.type = None
        self.score = self._score()
        self.target_count = 0
        self.assigned_armada = None
        self._occupied = False

    #---------------------------------------------------------------------------
    def occupied(self, is_occupied = True):
        self._occupied = is_occupied

    def is_occupied(self):
        return self._occupied


    #---------------------------------------------------------------------------
    def is_mutex(self):
        if(self.type == CellType.HALITE):
            return True
        elif(self.type == CellType.FRIENDLY_SHIP):
            return True
        else:
            return False

    #---------------------------------------------------------------------------
    def _score(self):
        shipyard = board.cells[self._position].shipyard
        ship = board.cells[self._position].ship

        if(shipyard != None and shipyard.player_id != me.id):
            self.type = CellType.ENEMY_SHIPYARD
            return SHIPYARD_HEAT_GRADE

        elif(ship != None and ship.player_id != me.id):
            self.type = CellType.ENEMY_SHIP
            halite_add = ship.halite if SHIP_USE_CARGO is True else 0
            return SHIP_HEAT_GRADE + halite_add

        else:
            self.type = CellType.HALITE
            return board.cells[self._position].halite

    #---------------------------------------------------------------------------
    def __str__(self) -> str:
        return str(self.score)

#--------------------------------------------------------------------------------
class HeatMap(object):
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
        return map_max(self.cells)
           
    #---------------------------------------------------------------------------
    def add_occupier(self, unit_id, position):
        self.cells[position].occupied()

    #---------------------------------------------------------------------------
    def is_occupied(self, position):
        return self.cells[position].is_occupied()

    #---------------------------------------------------------------------------
    def is_mutex(self, position):
        return self.cells[position].is_mutex()

    #---------------------------------------------------------------------------
    def create_opportunity_map(self, base_case, coeff_map):
        ''' Calculating opportunity cost by subtracting out how much halite we
            could have gathered if we stayed put rather than moved to a new location.
        '''
        result = {}
        for x in range(size): 
            for y in range(size): 
                pos = Point(x,y)
                cost = self.cells[pos].score - coeff_map[pos] * base_case
                result.update({pos : cost})
        return result

    #--------------------------------------------------------------------------------
    def __str__(self) -> str:
        return map_to_string(self.cells)
#end region

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
def angle_to_pos(fromPos, toPos):
    ''' Returns angle between two points
    '''
    fromX, fromY = fromPos[0], fromPos[1]
    toX, toY = toPos[0], toPos[1]
    if abs(fromX - toX) > size / 2:
        fromX += size
    if abs(fromY - toY) > size / 2:
        fromY += size

    return  math.degrees(math.atan2(toY - fromY, toX - fromX))


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
def map_max(map):  
     v=list(map.values())
     k=list(map.keys())
     max_v = max(v)
     return k[v.index(max_v)], max_v

#--------------------------------------------------------------------------------
def map_mult(map1, map2):  
    ''' Multiply two maps together '''
    #TODO: Should put checks in for key not existing in other map
    result = {}
    for key in map1.keys():
        result[key] = map1[key] * map2[key]
    return result

#--------------------------------------------------------------------------------
def map_add(map1, map2):  
    ''' Multiply two maps together '''
    #TODO: Should put checks in for key not existing in other map
    result = {}
    for key in map1.keys():
        result[key] = CLOSE_TO_HOME_MULTI * map1[key] + map2[key]
    return result
        
        
#--------------------------------------------------------------------------------
def get_moves_region(pos, depth = 2):
    ''' Returns a region (list of points) based on making 'depth' number of moves from a position.
        Useful for limiting area to X number of moves away.
    '''
    current = [pos] + get_neighbors(pos)
    destinations = current

    for i in range(1,depth):
        next_level = []
        for dest in current:
            candidates = [x for x in get_neighbors(dest) if x not in destinations]
            destinations = destinations + candidates
            next_level = next_level + candidates
        current = next_level
    return destinations

#--------------------------------------------------------------------------------
def get_area_region(pos, half_length):
    ''' Returns square area with side equal to 2 * half_length + 1
    '''
    result = []
    for i in range(2*half_length + 1):
        x = (pos.x - half_length) + i
        y = (pos.y - half_length) + i

        if(x < 0): x += size
        if(y < 0): y += size
        if(x >= size): x -= size
        if(y >= size): y -= size

        result.append(Point(x,y))

    return result

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
        total += HALITE_GATHER_RATE
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

    global me
    me = board.current_player

    global opponents
    opponents = board.opponents

    avg_halite = get_avg_halite()
    print("Average halite: " + str(avg_halite))


    #INITIALIZATION CODE
    #region INIT
    global initialized
    global coefficient_maps
    global opportunity_cost
    global starting_pos
    if not initialized:
        print("Initializing")
        init_coefficient_map(size)
        initialized = True
        starting_pos = me.ships[0].position
        
    #endregion

    #TEST CODE
    agent.counter = getattr(agent, 'counter', 0) + 1
    print("Turn " + str(agent.counter))


    # CONTROL HELPERS
    #region HELPER FUNCTIONS


    #--------------------------------------------------------------------------------
    def get_enemies_in_region(region, halite_thresh = 0):
        """ Return a list of all enemies in a region """
        enemies = []
        for pos in region:
            cell = board.cells[pos]
            if (cell.ship is not None and cell.ship.player_id != me.id and cell.ship.halite <= halite_thresh):
                enemies.append(cell.ship)
        return enemies

    #--------------------------------------------------------------------------------
    def flee_enemies(pos, enemies):
        """ Looks to see if enemies close to ship, if so we move in opposite directions """

        print("flee_enemies: " + str(len(enemies)) + " from " + str(pos))

        total = 0
        for en in enemies:
            print("flee_enemies: angle to " + str(en.position) + " is " + str(angle_to_pos(pos, en.position)))
            total += angle_to_pos(pos, en.position)

        avg_angle = total/len(enemies)
        print("flee_enemies: avg_angle = " + str(avg_angle))

        opposite = avg_angle + 180
        if(opposite > 360): 
            opposite -= 180

        print("flee_enemies: opposite = " + str(opposite))

        print("flee_enemies: " + str(move_from_angle(opposite)))
        return move_from_angle(opposite)

    #--------------------------------------------------------------------------------
    def move_from_angle(angle):
        if(angle < 45 and angle >= 315):
            return ShipAction.EAST
        elif(angle < 115 and angle >= 45):
            return ShipAction.NORTH
        elif(angle < 225 and angle >= 115):
            return ShipAction.WEST
        else:
            return ShipAction.SOUTH

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
            if not base_map.is_occupied(new_pos):
                base_map.add_occupier(ship.id, new_pos)
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
        spawn = False
        if(base_map.is_occupied(shipyard.position)):
            spawn = False
            print("Shipyard " + shipyard.id + " blocked from creating by ship sitting on it.")

        elif len(me.ships) < MAX_UNITS and me.halite >= 500:
            spawn = True

        elif check_region_for_enemy(get_neighbors(shipyard.position), 'SHIPYARD_DEFENSE'):
            print("Shipyard " + shipyard.id + " spawning defensive ship.")
            spawn = True

        if spawn:
            shipyard.next_action = ShipyardAction.SPAWN

        return

    #--------------------------------------------------------------------------------
    def ship_preprocess(meta, return_path):
        ''' Handles pre-processing steps where we want hardcoded actions for ships
            in specific situations.
        '''
        print("ship_preprocess: ship " + meta.id + " at: " + str(meta.ship.position) + " halite: " + str(meta.ship.halite))

        if len(me.shipyards) == 0:
            debug("ship_preprocess: Converting to shipyard")
            meta.ship.next_action = ShipAction.CONVERT
            return

        region = get_moves_region(meta.ship.position, SAFE_PROXIMITY)
        enemies = get_enemies_in_region(region, meta.ship.halite)
        if enemies:
            dir = flee_enemies(meta.ship.position, enemies)
            meta.ship.next_action = position_deconflict(meta.ship, dir)
            return

        #Return to base if ship has a lot of halite or Me(player) is low on halite and this ship has most halite
        if (meta.ship.halite > RETURN_HALITE_MIN) or (return_candidate is not None and meta.ship.id == return_candidate.id):
            shipyard = closest_shipyard(ship)
            meta.destination = shipyard.position
            print("ship_preprocess: " + ship.id + " is returning to base.  Returning to " + str(meta.destination))

        else:
            #TODO:  Move this function elsewhere
            meta.create_heat_map(base_map, avg_halite, coefficient_maps[ship.position], return_path)   
            to_rank.append(meta)

        meta_ships.append(meta)

        
    #--------------------------------------------------------------------------------
    def ship_execute(meta):
        ''' Final logic for translating meta actions into actual actions
        '''
        print("ship_execute for " + meta.id)

        if meta.ship.next_action is None:
            dir = get_direction_to(meta.ship.position, meta.destination)

            print("ship_execute for " + meta.id + " at: " + str(meta.ship.position) + " wants to go to: " + str(meta.destination))

            dir = position_deconflict(meta.ship, dir)              # Make sure we don't run into our own units

            if(dir is not None):
                meta.ship.next_action = dir
                print("ship_execute: next dir " + str(dir))

        return

    #--------------------------------------------------------------------------------
    def process_rank(rank_list, remove_point = None):
        ''' Iterative function to process a rank list and assign tasks based on highest
            ranking grade first (gets preference), working to lowest rank.

            remove_point is used if we want to remove a point from each rank_list element's
            heat map.  This will in effect make this point unchoosable by the rest of the list.
        '''

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

        #TODO: Goal here is to make halite mutually exclusive but enemy ships not, could change the logic and add a mutually
        #   exclusive flag, pushing the mutual exclusive decision into the class
        if base_map.is_mutex(max_meta.destination):
            process_rank(rank_list, max_meta.destination)   
        else:
            print("Processing without removing.")
            process_rank(rank_list)   

              
    #endregion


    #--------------------------------------------------------------------------------
    # Actual Function Code
    #--------------------------------------------------------------------------------
    base_map = HeatMap(size)
    print(board)
    
    print("Base map")
    print(base_map)

    # Calculate opportunity cost for returning haltie to a shipyard
    # TODO: Account for multiple shipyards
    return_path = coefficient_maps[Point(0,0)]  #TODO: Set some default
    if len(me.shipyards) > 0: 
        return_path = coefficient_maps[me.shipyards[0].position]

    #print("Return path")
    #print(map_to_string(return_path))
    
    # Set actions for each ship
    to_rank = []
    meta_ships = []

    #Figure out ship that is best candidate to return to base if we are low on halite
    return_candidate = None
    if me.halite < RETURN_HALITE_MIN: 
        return_candidate = max(me.ships, key = attrgetter('halite')) if len(me.ships) > 0 else None
        if return_candidate.halite < 500: return_candidate = None
        print("Return candidate: " + return_candidate.id + " with " + str(return_candidate.halite) + " halite.") if return_candidate is not None else print (" ")

    for ship in me.ships:
        meta = ShipMeta(ship)
        #TODO:  If had multiple shipyards, should figure out which one to use for return path here
        ship_preprocess(meta, return_path)
        
    process_rank(to_rank)

    for m in meta_ships:
        ship_execute(m)

    for shipyard in me.shipyards:
        shipyard_control(shipyard)

    return me.next_actions