# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        """
        features: ghost position, foods position, capsules position (of nextState)
        how to calculate score: 
            
        """
        ghostPos = successorGameState.getGhostPosition(1)
        foods = newFood.asList()
        capsules = currentGameState.getCapsules()
        score = successorGameState.getScore()

        ghostDist = manhattanDistance(newPos, ghostPos)
        score += max(ghostDist, 10)

        minDist = 1000
        for food in foods:
            dist = manhattanDistance(food, newPos)
            if dist < minDist:
                minDist = dist
        
        score += 50 / minDist
        if len(foods) < len(currentGameState.getFood().asList()):
            score += 100

        if newPos in capsules:
            score += 200

        return score





def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, 0, self.depth)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -1e9
        beta = 1e9
        return self.alpha_beta(gameState, 0, self.depth, alpha, beta)[1]

    def alpha_beta(self, state, agent_index, depth, alpha, beta):
        if state.isWin() or state.isLose() or depth == 0:
            return (self.evaluationFunction(state), 'Stop')

        agent_index %= state.getNumAgents()

        if agent_index == state.getNumAgents() - 1: # last ghost
            depth = depth - 1

        if agent_index == 0:
            return self.max_value(state, agent_index, depth, alpha, beta)
        else:
            return self.min_value(state, agent_index, depth, alpha, beta)

    def max_value(self, state, agent_index, depth, alpha, beta):
        actions = []
        value = -1e9
        ret_action = ''
        for action in state.getLegalActions(agent_index):
            successor_value = self.alpha_beta(
                state.generateSuccessor(agent_index, action),
                agent_index + 1, 
                depth,
                alpha, 
                beta
            )[0]

            actions.append((successor_value, action))
            #if value < successor_value:
            #    value = successor_value
            #    ret_action = action
            value = max(value, successor_value)
            ret_action = action

            if value > beta:
                return value, ret_action
            alpha = max(alpha, value)
        return max(actions) 

    def min_value(self, state, agent_index, depth, alpha, beta):
        actions = []
        value = 1e9
        ret_action = ''
        for action in state.getLegalActions(agent_index):
            successor_value = self.alpha_beta(
                state.generateSuccessor(agent_index, action),
                agent_index + 1, 
                depth,
                alpha, 
                beta
            )[0]

            actions.append((successor_value, action))

            value = max(value, successor_value)
            ret_action = action

            if value < alpha:
                return value, ret_action
            beta = min(beta, value)
        return min(actions) 

    def max_value(self, state, agent, depth):
        actions = []
        for action in state.getLegalActions(agent):
            actions.append(
                (
                    self.minimax(
                        state.generateSuccessor(
                            agent,
                            action
                        ), 
                        agent + 1, 
                        depth
                    )[0],
                    action
                )
            )
        return max(actions)

    def min_value(self, state, agent, depth):
        actions = []
        for action in state.getLegalActions(agent):
            actions.append(
                (
                    self.minimax(
                        state.generateSuccessor(
                            agent, 
                            action
                    ), 
                    agent + 1, 
                    depth)[0],
                    action
                )
            )
        return min(actions)


    def minimax(self, state, agent, depth, alpha, beta):
        if state.isWin() or state.isLose() or depth == 0:
            return (self.evaluationFunction(state), 'Stop')

        agent = agent % state.getNumAgents()
        if agent == state.getNumAgents() - 1:
            depth = depth - 1

        if agent == 0:
            return self.max_value(state, agent, depth, alpha, beta)
        else:
            return self.min_value(state, agent, depth, alpha, beta)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
