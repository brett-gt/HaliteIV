#SDK Reference: https://github.com/Kaggle/kaggle-environments/blob/39685f13f0e06ac18d4e6e09ac7c61c23db8658e/kaggle_environments/envs/halite/helpers.py#L347
# https://www.kaggle.com/superant/halite-boilerbot


# TODO:
#   Need to track individual assignments of units, if one is trying to destroy an enemy, dont let another take do the same
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

DEBUG = False

# Number of gatherers to maintain
NUMBER_OF_GATHERERS = 3
MAX_UNITS = 5

# How much more halite must be available elsewhere to bother moving off current spot
GATHER_MOVE_FACTOR = 1.2

# Treshhold beyond which a friendly unit returns to base
RETURN_HALITE_THRESH = 1000


HALITE_GATHER_RATE = 0.25
HALITE_REGEN_RATE  = 1.02

#--------------------------------------------------------------------------------
# Global Types
#--------------------------------------------------------------------------------
class ShipTask(Enum):
    NONE = 1
    GATHER = 2
    ATTACK = 3

#--------------------------------------------------------------------------------
def debug(s):
    if(DEBUG):
        print(s)

#--------------------------------------------------------------------------------
# Agent
#--------------------------------------------------------------------------------
def agent(obs,config):

    #TEST CODE
    agent.counter = getattr(agent, 'counter', 0) + 1
    print("Turn " + str(agent.counter))

    #Grab information up front
    size = config.size
    board = Board(obs,config)
    me = board.current_player
    opponents = board.opponents

    #Create blank map where we store our moves for deconfliction
    # TODO:  Make a class
    # next_map = [[None for i in range(size)] for j in range(size)]


    agent.fleet = getattr(agent, 'fleet', [])


    #GENERAL
    #region GENERAL


    #--------------------------------------------------------------------------------
    def argmax(arr, key=None):
        return arr[arr.index(max(arr, key=key)) if key else arr.index(max(arr))]

    #endregion


    # DISTANCE AND DIRECTION
    #region DISTANCE AND DIRECTION
    #--------------------------------------------------------------------------------
    def get_direction_to(fromPos, toPos, possList=[True, True, True, True]):
        ''' Returns best direction to move from one position (fromPos) to another (toPos)
        '''
        fromX, fromY = fromPos[0], fromPos[1]
        toX, toY = toPos[0], toPos[1]
        if abs(fromX-toX) > size / 2:
            fromX += size
        if abs(fromY-toY) > size / 2:
            fromY += size
        if fromY < toY and possList[0]: return ShipAction.NORTH
        if fromY > toY and possList[1]: return ShipAction.SOUTH
        if fromX < toX and possList[2]: return ShipAction.EAST
        if fromX > toX and possList[3]: return ShipAction.WEST
        return None

    #--------------------------------------------------------------------------------
    def get_linear_distance(pointA, pointB):
        ''' Returns best direction to move from one position (fromPos) to another (toPos)
        
            TODO: Reference Manhattan distance (https://www.kaggle.com/tmbond/halite-example-agents)
        '''
        x = abs(pointA.x - pointB.x)
        y = abs(pointA.y - pointB.y)
        return sqrt(x**2 + y**2)

    #--------------------------------------------------------------------------------
    def manhattan_distance(pos1, pos2):
        """Gets the Manhattan distance between two positions, i.e.,
        how many moves it would take a ship to move between them."""
        # TODO: Unit test
        dx = manhattan_distance_single(pos1.x, pos2.x)
        dy = manhattan_distance_single(pos1.y, pos2.y)
        return dx + dy

    #--------------------------------------------------------------------------------
    def manhattan_distance_single(i1, i2):
        """Gets the distance in one dimension between two columns or two rows, including wraparound."""
        # TODO: Unit test
        iMin = min(i1, i2)
        iMax = max(i1, i2)
        return min(iMax - iMin, iMin + size - iMax)

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

    #endregion

    # CONTROL HELPERS
    #region HELPER FUNCTIONS

    #--------------------------------------------------------------------------------
    def get_region(pos, depth = 2):
        ''' Returns a region based on making 'depth' number of moves from a position.

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
    def find_halite(ship, depth):
        ''' Find the maximum halite available assuming we want to make 'Depth' number
            of moves.  
        '''
        destinations = get_region(ship.position, depth)
        best = argmax([item.to_index(size) for item in destinations], key=obs.halite.__getitem__)
        return Point.from_index(best, size)

    #--------------------------------------------------------------------------------
    def halite_prediction(position, turns):
        ''' Return how much halite would be gathered by sitting on position for X turns
        '''
        current = board.cells[position].halite
        total = 0

        for i in range(turns):
            gathered = HALITE_GATHER_RATE * current
            total += gathered
            current = HALITE_REGEN_RATE * (current - gathered)

        return total

    #--------------------------------------------------------------------------------
    def closest_enemy(ship):
        ''' Find the closest enemy ship

            TODO: Add criteria for halite
        '''
        min_dist = 1000000
        closest = None
        for player in opponents:
            for enemy in player.ships:
                dist = manhattan_distance(ship.position, enemy.position)
                if(dist < min_dist):
                    closest = enemy
                    min_dist = dist

        return closest, min_dist
       
    #--------------------------------------------------------------------------------
    def closest_base(ship):
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
    class enemy_metadata(object):
        ''' Class to hold meta data about enemy units.
        '''
        def __init__(self, unit):
            pass;

    #--------------------------------------------------------------------------------
    class my_metadata(object):
        ''' Class to hold metadata about friendly ship actions.
            Desire is to make this static so I can track the state of the last actions.

            Trying to save actions by id, assuming those stay unique throughout entire run.
        '''
        def __init__(self, ship):
            debug("ship_metadata: Initializing new ship " + ship.id)
            self.id = ship.id
            self.action = None
            self.state = ShipTask.NONE
            self.blocked_timer = 0                #TODO: Could measure time blocked by other unit and vary actions      
            self.dist_to_closest_shipyard = 0     # Track for end of the game return
            self.update(ship)

        def update(self, ship):
            debug("ship_metadata: Updating " + self.id)
            self.position = ship.position

            closest, min_dist = closest_enemy(ship)
            self.closest_enemy = closest.id
            self.closest_dist  = min_dist

        def __repr__(self):
            return {'id':self.id, 'action':self.action}

        def __str__(self):
            result = "Ship: " + self.id + " current action: " + str(self.action) + "\n";
            result = result + "Closest enemy: " + self.closest_enemy + " is " + str(self.closest_dist) + " away.\n"
            return result;

    #--------------------------------------------------------------------------------
    class map(object):
        ''' Map class similar to the board object.  It looked like board object would
            be difficult to use for my intended purpose so creating a subset of it
        '''
        def __init__(self, size):
            self.size = size
            self.cells = [[None for i in range(size)] for j in range(size)]


        def add_unit(self, unit_id, position):
            self.cells[position.x][position.y] = unit_id

        def is_free(self, position):
            if(self.cells[position.x][position.y] == None):
                return True
            else:
                return False

        def __str__(self) -> str:
            '''
            Use same string method as the board.
            '''
            size = self.size
            result = ''
            for y in range(size):
                result += str(y).rjust(3)
                for x in range(size):
                    cell = self.cells[x][size - y - 1]

                    result += '|'
                    result += cell.rjust(4) if cell is not None else '    '
                    
                    # This normalizes a value from 0 to max_cell halite to a value from 0 to 9
                    #normalized_halite = int(9.0 * cell.halite / float(self.configuration.max_cell_halite))
                    #result += str(normalized_halite)
                    #result += (
                    #    chr(ord('A') + cell.shipyard.player_id)
                    #    if cell.shipyard is not None
                    #    else ' '
                    #)
                result += '|\n'

            result += ' '.rjust(4) 
            for x in range(size):
                result += str(x).center(5) 
            result += '\n'
            return result


    #--------------------------------------------------------------------------------
    def update_state():
        # Remove ships that no longer exist
        for ship in agent.fleet:
            if(ship.id not in me.ship_ids):
                debug("update_state: Removing " + ship.id)
                agent.fleet.remove(ship)

        # Update all existing ships
        for ship in me.ships:
            match = [x for x in agent.fleet if x.id == ship.id]  

            if(match):
                debug("update_state: Ship found " + ship.id)
                match[0].update(ship)
            else:
                debug("update_state: Ship not found " + ship.id)
                agent.fleet.append(my_metadata(ship))

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
            if next_map.is_free(new_pos):
                next_map.add_unit(ship.id, new_pos)
                debug("position_deconflict : wanted " + str(dir) + " ended up with " + str(choice))
                return choice;

        print("position_deconflict error")
        return None

    #--------------------------------------------------------------------------------
    def ship_control(ship):

        if len(board.current_player.shipyards) == 0:
            ship.next_action = ShipAction.CONVERT
        else:
            #closest, min_dist = closest_enemy(ship)
            #dir = getDirTo(ship.position, closest.position)
            #ship.next_action = dir

            # Make a decision on what to do
            if(ship.halite > RETURN_HALITE_THRESH):
                shipyard = closest_base(ship)
                dest = shipyard.position

            else:
                dest = find_halite(ship, 4)
                dest_amount = board.cells[dest].halite

                dist = manhattan_distance(ship.position, dest)
                here_amount = halite_prediction(ship.position, dist)

                debug("ship_control: Gather target " + str(dest) + "=" + str(dest_amount) + " staying here: " + str(here_amount))

                if(dest_amount < GATHER_MOVE_FACTOR * here_amount):
                    dest = ship.position

            dir = get_direction_to(ship.position, dest)
            dir = position_deconflict(ship, dir)              # Make sure we don't run into our own units

            if(dir is not None):
                ship.next_action = dir


    #--------------------------------------------------------------------------------
    def shipyard_control(shipyard):

        # TODO: Needs updating for multiple shipyards

        # If there are no ships, use first shipyard to spawn a ship.
        if len(me.ships) < MAX_UNITS:
            shipyard.next_action = ShipyardAction.SPAWN


    #--------------------------------------------------------------------------------
    def assign_task():
        ''' Contains the logic for assigning task to individual ships
        '''
        for ship in agent.fleet:
            ship.task = ShipTask.GATHER

           
    #endregion


    #--------------------------------------------------------------------------------
    # Actual Function Code
    #--------------------------------------------------------------------------------
    next_map = map(size)
 
    update_state()

    assign_task()


    # Set actions for each ship
    for ship in me.ships:
        halite_prediction(ship.position, 3)
        ship_control(ship)

    debug(next_map)

    # Set actions for each shipyard
    for shipyard in me.shipyards:
        shipyard_control(shipyard)
    
    return me.next_actions