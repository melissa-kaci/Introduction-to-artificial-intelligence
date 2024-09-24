import numpy as np
import math
from pacman_module.game import Agent, Directions, manhattanDistance


class BeliefStateAgent(Agent):
    """Belief state agent.

    Arguments:
        ghost: The type of ghost (as a string).
    """

    def __init__(self, ghost):
        super().__init__()

        # Store the type of ghost
        self.ghost = ghost
        # Parameter n for the binomial distribution (nb of trials)
        self.n = 4
        # Parameter p pour the binomial distribution (prob of success)
        self.p = 0.5

    def transition_matrix(self, walls, position):
        """Builds the transition matrix

            T_t = P(X_t | X_{t-1})

        given the current Pacman position.
        It represents the probabilities of a ghost moving
        from one position to another.

        Arguments:
            walls: The W x H grid of walls.
            position: The current position of Pacman.

        Returns:
            The W x H x W x H transition matrix T_t. The element (i, j, k, l)
            of T_t is the probability P(X_t = (k, l) | X_{t-1} = (i, j)) for
            the ghost to move from (i, j) to (k, l).
        """
        # Initialization of variables to define the grid dimensions
        # + transition matrix
        width = walls.width
        height = walls.height

        transition_matrix = np.zeros((width, height, width, height))

        # Exponent factors based on the type of ghost
        fear_exponent = {
            'afraid': 1,  # 2^1 = 2
            'fearless': 0,  # 2^0 = 1
            'terrified': 3  # 2^3 = 8
                         }

        # Fear_factor based on the type of ghost
        fear_factor = fear_exponent.get(self.ghost, 0)

        # Loop through all positions on the grid
        for i in range(width):
            for j in range(height):

                # If the position is a wall => skip to the next iteration
                if walls[i][j]:
                    continue

                current_pos = (i, j)
                possible_moves = []
                distances = []

                # Check possible moves from the current position
                for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)]:  # EWNS
                    ni, nj = i + di, j + dj

                    # To be sure that the move stays within the grid
                    if ((0 <= ni <
                         width) and (0 <= nj < height) and not walls[ni][nj]):
                        possible_moves.append((ni, nj))
                        distances.append(manhattanDistance(position, (ni, nj)))
                # If no possible moves => ghost stays in place with proba = 1
                if not possible_moves:
                    transition_matrix[i, j, i, j] = 1.0
                    continue

                current_distance = manhattanDistance(position, current_pos)
                total_prob = 0

                # Assign probabilities to each possible move
                for move, dist in zip(possible_moves, distances):
                    ni, nj = move
                    # If distance from pacman increases => fear factor
                    if dist >= current_distance:
                        prob = 2**fear_factor
                    else:
                        prob = 1
                    transition_matrix[i, j, ni, nj] = prob
                    total_prob += prob

                # Normalization
                if total_prob > 0:
                    transition_matrix[i, j, :, :] /= total_prob

        return transition_matrix

    def observation_matrix(self, walls, evidence, position):
        """Builds the observation matrix

            O_t = P(e_t | X_t)

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The W x H observation matrix O_t.
        """

        # Initialization of the observation matrix
        observation_matrix = np.zeros((walls.width, walls.height))

        # Loop through all positions on the grid
        for i in range(walls.width):
            for j in range(walls.height):

                # If position is a wall => skip to the next iteration
                if walls[i][j]:
                    continue

                # Distance between Pacmand and (i,j) position
                dist = manhattanDistance(position, (i, j))

                # To take into account the noise
                z = evidence + self.n*self.p - dist

                # Checking if z is in a valid range of the bin distribution
                if 0 <= z <= self.n:
                    # Binomial probability stored in the matrix
                    binom_prob = math.comb(self.n, int(z)) * (self.p ** self.n)
                    observation_matrix[i, j] = binom_prob
        return observation_matrix

    def update(self, walls, belief, evidence, position):
        """Updates the previous ghost belief state

            b_{t-1} = P(X_{t-1} | e_{1:t-1})

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            belief: The belief state for the previous ghost position b_{t-1}.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The updated ghost belief state b_t as a W x H matrix.
        """
        # Compute transition and observation matrix
        T = self.transition_matrix(walls, position)
        O = self.observation_matrix(walls, evidence, position)

        # Dimensions of the grid
        width = walls.width
        height = walls.height

        # Formule : b_t = alpha x O x T x b_(t-1)

        # Reshape into a 1D array for the multiplication
        belief_reshaped = belief.flatten()

        # Reshape into a 2D matrix for the multiplication
        T_reshaped = T.reshape((width * height, width * height))

        # Compute the intermediate product T_t*b_(t-1)
        intermediate_belief = T_reshaped @ belief_reshaped  # [W*H]

        # Back to original shape
        intermediate_belief = intermediate_belief.reshape((width, height))

        # intermediaite * O
        new_belief = O*intermediate_belief

        # Normalization
        sum_belief = np.sum(new_belief)
        if sum_belief > 0:
            new_belief /= sum_belief

        return new_belief

    def get_action(self, state):
        """Updates the previous belief states given the current state.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            The list of updated belief states.
        """

        walls = state.getWalls()
        beliefs = state.getGhostBeliefStates()
        eaten = state.getGhostEaten()
        evidences = state.getGhostNoisyDistances()
        position = state.getPacmanPosition()

        new_beliefs = [None] * len(beliefs)

        for i in range(len(beliefs)):
            if eaten[i]:
                new_beliefs[i] = np.zeros_like(beliefs[i])
            else:
                new_beliefs[i] = self.update(
                    walls,
                    beliefs[i],
                    evidences[i],
                    position,
                )

        return new_beliefs


class PacmanAgent(Agent):
    """Pacman agent that tries to eat ghosts given belief states."""

    def __init__(self):
        super().__init__()

        # Ghost target for Pacman
        self.target = None
        # Pacman last position
        self.last_position = None

    def should_retarget(self, position, belief):
        """
        Checks if Pacman should recalculate its target

        Arguments:
            position: Pacman position
            belief: belief matrix for ghost target

        Returns:
            bool: true if pacman has to recalculate its target, false otherwise
        """
        # 1 => retarget if pacman hasn't moved
        if self.last_position == position:
            return True

        # 2 => retarget if current belief is very low
        max_belief = np.max(belief)
        if max_belief < 0.1:
            return True

        return False

    def _get_action(self, walls, beliefs, eaten, position):
        """
        Arguments:
            walls: The W x H grid of walls.
            beliefs: The list of current ghost belief states.
            eaten: A list of booleans indicating which ghosts have been eaten.
            position: The current position of Pacman.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        # Check if pacman needs to retarget => if no current target,
        # if ghost has been killed of should_retarget true
        if (
            self.target is None or
            eaten[self.target] or
            self.should_retarget(position, beliefs[self.target])
        ):
            max_prob = -1

            # Loop through all ghosts => the aim is to find
            # the one with the highest belief prob
            for i, belief in enumerate(beliefs):

                # Skip if the ghost has already been eaten
                if eaten[i]:
                    continue
                current_max_prob = np.max(belief)

            # If the ghost has the highest prob => update target
                if current_max_prob > max_prob:
                    max_prob = current_max_prob
                    self.target = i

        # Most likely position of the target ghost
        most_likely_position = np.unravel_index(np.argmax(
            beliefs[self.target]), beliefs[self.target].shape)

        # A* to find the optimal path to the most likely position of the ghost
        best_path = self.astar(position, most_likely_position, walls)

        if best_path and len(best_path) > 1:
            next_position = best_path[1]

            # Update pacmans last position
            self.last_position = next_position

            # Directions
            if next_position[0] > position[0]:
                return Directions.EAST
            elif next_position[0] < position[0]:
                return Directions.WEST
            elif next_position[1] > position[1]:
                return Directions.NORTH
            elif next_position[1] < position[1]:
                return Directions.SOUTH

        return Directions.STOP

    def astar(self, start, goal, walls):
        """ A* algorithm to find the optimal path

        Arguments:
            start : current position of pacman
            goal : goal position (most likely position of the ghost)
            walls : W x H grid
        """

        from queue import PriorityQueue

        # Initialisation of the fringe
        fringe = PriorityQueue()
        fringe.put((0, start))

        # Dict to keep track of the paths + their cost
        came_from = {start: None}
        cost_so_far = {start: 0}

        while not fringe.empty():
            # Position w/ the lowest priority
            _, current = fringe.get()

            if current == goal:
                break

            # All possible moves from current position
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_move = (current[0] + dx, current[1] + dy)

                # Skip if walls
                if walls[next_move[0]][next_move[1]]:
                    continue

                # New cost to reach position
                new_cost = cost_so_far[current] + 1

                # Update if cheaper
                if ((next_move not in
                     cost_so_far) or (new_cost < cost_so_far[next_move])):
                    cost_so_far[next_move] = new_cost
                    priority = new_cost + manhattanDistance(goal, next_move)
                    fringe.put((priority, next_move))

                    # Besth path
                    came_from[next_move] = current

        if current != goal:
            return []

        # Construction of the path
        path = []
        while current:
            path.append(current)
            current = came_from[current]
        # bc is built from goal to start
        path.reverse()
        return path

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        return self._get_action(
            state.getWalls(),
            state.getGhostBeliefStates(),
            state.getGhostEaten(),
            state.getPacmanPosition(),
        )
