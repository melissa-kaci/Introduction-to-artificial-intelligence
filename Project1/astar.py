from pacman_module.game import Agent, Directions
from pacman_module.util import PriorityQueueWithFunction
from pacman_module.util import manhattanDistance


def key(state):
    """Returns a key that uniquely identifies a Pacman game state.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.

    Returns:
        A hashable key tuple.
    """
    return (
        state.getPacmanPosition(),
        state.getFood()
    )


def aStarPriority(item):
    """Returns the priority associated with a Pacman game state.

    Arguments :
        item[3] : the backward cost associated with this state.
        item[4] : the forward cost associated with this state.

    Returns :
        An integer that is the estimated cost of cheapest solution.
        i.e. sum of the backward and forward cost (heuristic)
    """
    return item[3] + item[4]


def pathCost(state, capsules):
    """Returns the path cost and the updated list of capsules associated
    with a successor of a state.

    Arguments :
        state : a game successor of the current state.
        capsules : a list of positions of the remaining capsules in the grid
        for the current state.

    Returns :
        A tuple containing the path cost and the updated list of positions.
    """
    successorCapsules = state.getCapsules()
    if (len(successorCapsules) == len(capsules)):
        return (1, capsules)
    else:
        return (5, successorCapsules)


def heuristic(state):
    """Computes the value of the manhattan distance between Pacman
    and the food farthest from it.

    Arguments :
        state: a game state.

    Returns :
        An integer that is the forward cost for the given state.
    """
    currentFood = state.getFood()
    (x, y) = state.getPacmanPosition()

    distance = []

    for i in range(currentFood.width):
        for j in range(currentFood.height):
            if currentFood[i][j] is True:
                distance.append(manhattanDistance((i, j), (x, y)))

    if not len(distance):
        return 0
    else:
        return max(distance)


class PacmanAgent(Agent):
    """Pacman agent based on Astar (A*)."""

    def __init__(self):
        super().__init__()

        self.moves = None

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """
        if self.moves is None:
            self.moves = self.astar(state)

        if self.moves:
            return self.moves.pop(0)
        else:
            return Directions.STOP

    def astar(self, state):
        """Given a Pacman game state, returns a list of legal moves to solve
            the search layout.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A list of legal moves.
        """
        path = []
        capsules = state.getCapsules()
        fringe = PriorityQueueWithFunction(aStarPriority)
        fringe.push((state, path, capsules, 0, heuristic(state)))
        closed = set()

        while True:
            if fringe.isEmpty():
                return []

            priority, (current, path, capsules, backwardCost,
                       forwardCost) = fringe.pop()

            if current.isWin():
                return path

            current_key = key(current)

            if current_key in closed:
                continue
            else:
                closed.add(current_key)

            for successor, action in current.generatePacmanSuccessors():
                (cost, updatedCapsules) = pathCost(successor, capsules)
                fringe.push((successor, path + [action], updatedCapsules,
                             backwardCost + cost, heuristic(successor)))

        return path
