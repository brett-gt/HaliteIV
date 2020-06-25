#Training agent: https://www.kaggle.com/tmbond/halite-example-agents

from kaggle_environments.envs.halite.helpers import *
import sys
import traceback
import copy
import math
import pprint
from random import choice, randint, shuffle

DIRS = ["NORTH", "SOUTH", "EAST", "WEST"]

# Each ship id will be assigned a state, one of COLLECT or DEPOSIT, which decides what it will do on a turn.
states = {}

COLLECT = "collect"
DEPOSIT = "deposit"


def argmax(arr, key=None):
    return arr.index(max(arr, key=key)) if key else arr.index(max(arr))


# This function will not hold up in practice
# E.g. cell getAdjacent(224) includes position 0, which is not adjacent
def getAdjacent(pos):return [
    (pos - 15) % 225,
    (pos + 15) % 225,
    (pos +  1) % 225,
    (pos -  1) % 225
  ]

def getDirTo(fromPos, toPos):
    fromY, fromX = divmod(fromPos, 15)
    toY,   toX   = divmod(toPos,   15)

    if fromY < toY: return "SOUTH"
    if fromY > toY: return "NORTH"
    if fromX < toX: return "EAST"
    if fromX > toX: return "WEST"

    
def agent(obs, config):
    action = {}
    player_halite, shipyards, ships = obs.players[obs.player]

    for uid, shipyard in shipyards.items():
        # Maintain one ship always
        if len(ships) == 0:
            action[uid] = "SPAWN"

    for uid, ship in ships.items():
        # Maintain one shipyard always
        if len(shipyards) == 0:
            action[uid] = "CONVERT"
            continue

        # If a ship was just made
        if uid not in states: states[uid] = COLLECT

        pos, halite = ship

        if states[uid] == COLLECT:
            if halite > 2500:
                states[uid] = DEPOSIT

            elif obs.halite[pos] < 100:
                best = argmax(getAdjacent(pos), key=obs.halite.__getitem__)
                action[uid] = DIRS[best]

        if states[uid] == DEPOSIT:
            if halite < 200: states[uid] = COLLECT

            direction = getDirTo(pos, list(shipyards.values())[0])
            if direction: action[uid] = direction
            else: states[uid] = COLLECT

    return action
