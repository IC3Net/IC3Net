import math

def get_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def get_closest(unit, state, player_id):
    """Get the closest unit of player with 'player_id' for 'unit' passed

    Arguments:
        unit {tc.Unit} -- Unit for which closest unit is to be found
        state {tc.State} -- State from tc client
        player_id {number} -- Player id whose units are to be scanned for closest
    """

    opp_units = state.units[player_id]

    closest = None
    mini = math.inf
    for opp_unit in opp_units:
        dist = get_distance(unit.x, unit.y, opp_unit.x, opp_unit.y)
        if mini > dist:
            closest = opp_unit
            mini = dist
    return closest

def get_weakest(unit, state, player_id):
    opp_units = state.units[player_id]

    weakest = None
    mini = math.inf
    for opp_unit in opp_units:
        # This need to be absolute hitpoints not relative
        health = opp_unit.health + opp_unit.shield
        if mini > health:
            weakest = opp_unit
            mini = health
    return weakest