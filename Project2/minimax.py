from pacman_module.game import Agent, Directions
import numpy as np


def key(state):
    """Returns a unique key that identifies a pacman game state

    Arguments :
        state : a game state. See API or class 'pacman.GameState'

    Returns :
        A hashable key tuple to keep track of visited states and avoid cycles
        """
    return (
        state.getPacmanPosition(),
        state.getFood(),
        state.getGhostPosition(1),
        state.getGhostDirection(1)
    )


def terminalTest(state):
    """Returns true if the state is terminal (win or lose) """
    return state.isWin() or state.isLose()


class PacmanAgent(Agent):
    """ Pacman agent using minimax algorithm with alpha-beta pruning """

    def __init__(self):
        super().__init__()

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        alpha = -np.inf
        beta = np.inf
        # closed set is initialized for each decision step
        closed = set()

        # Minimax search
        move = self.minimax(state, 0, closed, alpha, beta)[0]

        return move if move else Directions.STOP

    def minimax(self, state, player, closed, alpha, beta):
        """Given a Pacman game state, searchs which action is the best to
          - maximize the score if it is Pacman's turn, or
          - minimize it if it is a Ghost's turn.

        Arguments:
            state: a game state.
            player: a value that defines which player's turn it is
                      - is equals to 0 for Pacman agent
                      - is 1 for Ghost agent
            closed: set of visited states to avoid revisiting
            alpha : best value that the maximizer can guarantee
            beta : best value that the minimizer can guarantee

        Returns:
            The best legal move for Pacman (at the given state)
            and the score (minimax value) associated to it
        """

        currentKey = key(state)

        # To avoid cycles
        if currentKey in closed:
            return None, state.getScore()

        # Add current state to closed before exploring successors
        closed.add(currentKey)

        # Chek if game is won or lost
        if terminalTest(state):
            return None, state.getScore()

        # Pacman's turn (MAX player)
        if player == 0:
            minimaxValue = -np.inf
            bestAction = None
            for successor, action in state.generatePacmanSuccessors():
                if action in state.getLegalActions(0):
                    score = self.minimax(successor, 1, closed, alpha, beta)[1]
                    if score > minimaxValue:
                        minimaxValue = score
                        bestAction = action
                    alpha = max(alpha, minimaxValue)
                    if beta <= alpha:
                        break  # beta cutoff

        # Ghost's turn (MIN player)
        elif player == 1:
            minimaxValue = +np.inf
            bestAction = None
            for successor, action in state.generateGhostSuccessors(player):
                if action in state.getLegalActions(1):
                    score = self.minimax(successor, 0,
                                         closed, alpha, beta)[1]
                    if score < minimaxValue:
                        minimaxValue = score
                        bestAction = action
                    beta = min(beta, minimaxValue)
                    if beta <= alpha:
                        break  # alpha cutoff

        return bestAction, minimaxValue
