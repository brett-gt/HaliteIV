    #--------------------------------------------------------------------------------
    def get_safe_moves(pos, my_ship_halite):
        ''' Loop through possible moves and determine which are likely to be safe 
            based on proximity of enemies
        '''
        #TODO: Candidate for removal

        move_list = ShipAction.moves() 
        good_moves = []

        region = get_moves_region(pos, SAFE_PROXIMITY)
        if check_region_for_enemy(region, my_ship_halite):
            good_moves.append(None)
                
        for move in move_list:
            new_pos = get_new_pos(pos, move)
            region = get_moves_region(new_pos, SAFE_PROXIMITY)
            if check_region_for_enemy(region, my_ship_halite, task = 'AVOID'):
                good_moves.append(move)
        return
  


#--------------------------------------------------------------------------------
    class map(object):
        ''' Map class similar to the board object.  It looked like board object would
        be difficult to use for my intended purpose so creating a subset of it
        
        '''
        def __init__(self, size):
            self.size = size
            self._cells: Dict[Point, Cell] = {}

            # Create a cell for every point in a size x size grid
            for x in range(size):
                for y in range(size):
                    position = Point(x, y)
                    halite = observation.halite[position.to_index(size)]
                    # We'll populate the cell's ships and shipyards in _add_ship and _add_shipyard
                    self.cells[position] = Cell(position, halite, None, None, self)


        def __str__(self) -> str:
            '''
            Use same string method as the board.
            '''
            size = self.configuration.size
            result = ''
            for y in range(size):
                for x in range(size):
                    cell = self[(x, size - y - 1)]
                    result += '|'
                    result += (
                        chr(ord('a') + cell.ship.player_id)
                        if cell.ship is not None
                        else ' '
                    )
                    # This normalizes a value from 0 to max_cell halite to a value from 0 to 9
                    normalized_halite = int(9.0 * cell.halite / float(self.configuration.max_cell_halite))
                    result += str(normalized_halite)
                    result += (
                        chr(ord('A') + cell.shipyard.player_id)
                        if cell.shipyard is not None
                        else ' '
                    )
                result += '|\n'
            return result

