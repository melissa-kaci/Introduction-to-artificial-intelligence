from pacman_module.game import Agent, Directions
from pacman_module.util import manhattanDistance
import numpy as np

# Constant for minimum distance between Pacman and the ghost
MINDISTANCE = 3


def key(state):
    """Returns a key that uniquely identifies a Pacman game state.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.

    Returns:
        A hashable key tuple to keep track of visited states and avoid cycles
    """
    return (
        state.getPacmanPosition(),
        state.getFood(),
        state.getGhostPosition(1),
        state.getGhostDirection(1)
    )


def cutOffTest(state, depth):
    """Given depth, decides if we stop expanding the current state.

    Arguments:
        state: a game state.
        depth: the current depth of recursion call

    Returns:
        A boolean value : true if the search should be stopped, false otherwise
    """

    # Stop the search if Pacman wins or losses
    if state.isWin() or state.isLose():
        return True

    # Max depth for the search
    MAXDEPTH = 1
    dynamicMaxDepth = MAXDEPTH

    # Maxdepth increases if there are maximum two foods remaining
    if state.getNumFood() <= 2:
        dynamicMaxDepth += 1

    pacmanPosition = state.getPacmanPosition()
    ghostPosition = state.getGhostPosition(1)

    # Distance between Pacman and the ghost
    ghostDistance = manhattanDistance(pacmanPosition, ghostPosition)

    # Depth increases if the ghost is far away
    if ghostDistance >= 5:
        dynamicMaxDepth += 1

    # Stop search
    if depth >= dynamicMaxDepth:
        return True


class PacmanAgent(Agent):
    """ Pacman agent using hminimax + alpha beta pruning """

    def __init__(self):
        super().__init__()
        # Set to store the keys of visited states
        self.closed = set()
        # Dictionary with the number of times each state is visited
        self.counter = {}

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        currentKey = key(state)
        hminimaxValue = -np.inf
        bestAction = Directions.STOP

        # To avoid cycles
        if currentKey in self.closed:
            visitCount = self.counter.get(currentKey, 0)
            self.counter[currentKey] = visitCount + 1
        else:
            # Mark a state as visited and add a visit
            self.closed.add(currentKey)
            self.counter[currentKey] = 1

        # Choosing the best action for each successor
        for successor, action in state.generatePacmanSuccessors():
            _, score = self.hminimax(successor, 1, 0, -np.inf,
                                     np.inf, currentKey)
            if score > hminimaxValue:
                hminimaxValue = score
                bestAction = action
        return bestAction

    def eval(self, state, evalKey):
        """Given a Pacman game state, estimates its expected utility

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A numerical value of the hminimax score, estimated with game state
            and parameters used to compute score function [See INSTRUCTIONS]
        """
        currentKey = key(state)
        pacmanPosition = currentKey[0]
        currentFood = currentKey[1]
        ghostPosition = currentKey[2]

        # Distance to the ghost + penalty if pacman is too close
        ghostDistance = manhattanDistance(pacmanPosition, ghostPosition)
        ghostPenalty = 0
        if ghostDistance < MINDISTANCE:
            ghostPenalty = MINDISTANCE - ghostDistance
        else:
            ghostPenalty = 0

        # Score based on the distance to the nearest food dot
        score = 0
        foodList = currentFood.asList()
        foodScore = 0

        # If one dot => calculate the distance to it
        if len(foodList) == 1:
            foodScore = -manhattanDistance(pacmanPosition, foodList[0])
        else:
            # If more than one dot => shortest path
            while len(foodList) > 1:
                distance_list = [manhattanDistance(food, pacmanPosition)
                                 for food in foodList]
                nearest_food_distance = min(distance_list)
                current_nearest_food = foodList.pop(
                    distance_list.index(nearest_food_distance))
                pacmanPosition = current_nearest_food
                score -= nearest_food_distance
            foodScore = score

        # Penalty for revisiting states
        counterValue = self.counter.get(evalKey, 0)

        # Ponderate evaluation score
        scoreEval = (state.getScore() + 1.5*foodScore - counterValue
                     - ghostPenalty + 0.5*ghostDistance)

        return scoreEval

    def hminimax(self, state, player, depth, alpha, beta, currentKey):
        """Given a Pacman game state, searchs which action is the best to
          - maximizes the score if it is Pacman's turn, or
          - minimizes it if it is a Ghost's turn.

        Arguments:
            state: a game state.
            player: a value that defines which player's turn it is
                      - is equals to 0 for Pacman agent
                      - is 1 for Ghost agent
            alpha and beta: values for pruning
            depth: the current depth of recursion call
            currentKey: a game state key

        Returns:
            The score (hminimax value) associated to the best legal move for
            Pacman (for the given state), according to minimax algorithm
        """
        # If cutoff test met => evaluation of the current state
        if cutOffTest(state, depth):
            return None, self.eval(state, currentKey)

        bestAction = None

        # Pacman's turn (MAX)
        if player == 0:
            hminimaxValue = -np.inf
            for successor, action in state.generatePacmanSuccessors():
                _, score = self.hminimax(successor, 1, depth + 1, alpha, beta,
                                         currentKey)
                if (score > hminimaxValue):
                    hminimaxValue = score
                    bestAction = action
                alpha = max(alpha, hminimaxValue)
                if hminimaxValue >= beta:
                    break
            return bestAction, hminimaxValue

        # Ghost's turn (MIN)
        elif player == 1:
            hminimaxValue = +np.inf
            for successor, action in state.generateGhostSuccessors(player):
                _, score = self.hminimax(successor, 0, depth + 1, alpha, beta,
                                         currentKey)
                if (score < hminimaxValue):
                    hminimaxValue = score
                    bestAction = action
                beta = min(beta, hminimaxValue)
                if hminimaxValue <= alpha:
                    break
        return bestAction, hminimaxValue
