# -*- coding: utf-8 -*-

import numpy as np
import random
from a_star_numbergrid import astar
import matplotlib.pyplot as plt

class gridGeneratorEnv():
    """Custom Environment that follows gym interface"""
    def __init__(self, maze_size = (4, 4), screen_size=(500, 500)):
        
        self.maze_size = maze_size
        
        # list of all cell locations
        self.grid = np.ones(self.maze_size)
        self.maze_withwalls = self.generateInitialGrid()
        print(self.maze_withwalls)
        self.actionSpace = {'U': -self.MAZE_W, 'D': self.MAZE_W, 'L': -1, 'R': 1}
        self.possibleActions = ['U', 'D', 'L', 'R']

        
        self.stateSpace = [i for i in range(self.MAZE_W*self.MAZE_H)]

        #State space for walls
        self.stateSpaceWalls = [i for i in range(self.WALLMAZE_W*self.WALLMAZE_H)]
        
        #Remove states corresponding to corners
        cornerloc = []
        for i in np.argwhere(self.maze_withwalls == 0):
            cornerloc.append(self.WALLMAZE_W*i[0] + i[1])
        self.stateSpaceWalls = [ele for ele in self.stateSpaceWalls if ele not in cornerloc]        
        
        #The four directions in which walls can be placed
        self.wallactionSpace = {'U': -self.WALLMAZE_W, 'D': self.WALLMAZE_W,
                            'L': -1, 'R': 1}
        
        #Setting tuples for Start, Exit and Treasure cells
        self.startXY = (0,0) #Top-left corner
        self.exitXY = (self.WALLMAZE_W-1, self.WALLMAZE_H-1) #Bottom-right corner
        self.treasureXY = (2*random.randint(0,self.MAZE_W-1), 2*random.randint(0, self.MAZE_H-1))
        


    def isTerminalState(self, state):
        return state == self.exit
    
    #Find the x,y location of the agent from its state position
    def getAgentRowAndColumn(self):
        x = self.agentPosition // self.MAZE_W
        y = self.agentPosition % self.MAZE_H
        return x, y
    
    #Find the x,y location of the agent when walls are included, from its state position
    def getAgentRowAndColumnWithWalls(self):
        x = (self.agentPosition // self.MAZE_W)*2
        y = (self.agentPosition % self.MAZE_H)*2
        return x,y
    
    def getXYCoordinatesWalls(self, state):
        x = state // self.WALLMAZE_W
        y = state % self.WALLMAZE_H
        return (x,y)
    
    def setState(self, state):
        x, y = self.getAgentRowAndColumn()
        self.grid[x][y] = 0
        self.agentPosition = state
        x, y = self.getAgentRowAndColumn()
        self.grid[x][y] = 1
        
    def generateInitialGrid(self):
        m = np.ones((self.WALLMAZE_W, self.WALLMAZE_H))
        n = m[1::2]
        for i in n:
            i[1::2]=0
        self.maze_withwalls = m
        return self.maze_withwalls
    
    def getGridAfterWallPlacing(self, wall_action):
        x, y = self.getAgentRowAndColumnWithWalls()
        self.maze_withwalls[x][y]


    def offGridMove(self, newState, oldState):
        # if we move into a row not in the grid
        if newState not in self.stateSpace:
            return True
        # if we're trying to wrap around to next row
        elif oldState % self.MAZE_W == 0 and newState  % self.MAZE_W == self.MAZE_W - 1:
            return True
        elif oldState % self.MAZE_W == self.MAZE_W - 1 and newState % self.MAZE_W == 0:
            return True
        else:
            return False
        
    #Function to check if the wall being built is outside the grid
    def offGridMoveWall(self, newState, oldState):
        # if we move into a row not in the grid
        if newState not in self.stateSpaceWalls:
            return True
        # if we're trying to wrap around to next row
        elif oldState % self.WALLMAZE_W == 0 and newState  % self.WALLMAZE_W == self.WALLMAZE_W - 1:
            return True
        elif oldState % self.WALLMAZE_W == self.WALLMAZE_W - 1 and newState % self.WALLMAZE_W == 0:
            return True
        else:
            return False

    def step(self, action, wallaction):
        done_flag = False
        
        #Get agent X and Y positions
        agentX, agentY = self.getAgentRowAndColumn()
        
        #X and Y positions when walls are considered as cells
        agentXwithWalls, agentYwithWalls = self.getAgentRowAndColumnWithWalls()
        
        #Agent position in wall state space
        self.agentPositionWithWalls = self.WALLMAZE_W*agentXwithWalls + agentYwithWalls

        #Finding the resulting state after action
        resultingState = self.agentPosition + self.actionSpace[action]
        
        #Finding the wall that will be built
        resultingWallState = self.agentPositionWithWalls + self.wallactionSpace[wallaction]
        addWallXY = self.getXYCoordinatesWalls(resultingWallState)

        #Making sure that the wall to be built is not off grid
        if not self.offGridMoveWall(resultingWallState, self.agentPositionWithWalls):
            self.maze_withwalls[addWallXY[0]][addWallXY[1]] = 0
            if astar(self.maze_withwalls, self.startXY, self.exitXY):
                reward_walls = 1
            else:
                reward_walls = -10
                done_flag = True
                
            if astar(self.maze_withwalls, self.startXY, self.treasureXY):
                reward_walls = 1
            else:
                reward_walls = -10
                done_flag = True
        else:
            reward_walls = 0
            
        reward = 0 if not self.isTerminalState(resultingState) else 0
                
        if not self.offGridMove(resultingState, self.agentPosition):
            self.setState(resultingState)
            return resultingState, resultingWallState, reward, reward_walls, \
                   self.isTerminalState(resultingState) and done_flag, None
        else:
            return self.agentPosition, self.agentPositionWithWalls, reward, reward_walls, \
                   self.isTerminalState(self.agentPosition) and done_flag, None

    def reset(self):
        self.grid = np.ones(self.maze_size)
        self.maze_withwalls = self.generateInitialGrid()
        
        self.start = 0
        self.startXY = (0,0)
                
        self.exit = self.WALLMAZE_W*self.WALLMAZE_H
        self.exitXY = (self.WALLMAZE_W-1, self.WALLMAZE_H-1)
        print(self.exitXY)
        self.treasureXY = (2*random.randint(0,self.MAZE_W-1), 2*random.randint(0, self.MAZE_H-1))
        while (self.treasureXY == self.startXY) and (self.treasureXY == self.exitXY):
            self.treasureXY = (2*random.randint(0,self.MAZE_W-1), 2*random.randint(0, self.MAZE_H-1))
        print(self.treasureXY)

#        random.choice(self.stateSpace)
#        while ((self.exit == self.start) and (self.exit == self.treasure)):
#            self.exit = random.choice(self.stateSpace)
#        self.exitXY = self.getXYCoordinatesWalls(self.exit)
        
        self.agentPosition = self.start
        self.agentPositionWithWalls = self.start
        return self.agentPosition, self.agentPositionWithWalls  # reward, done, info can't be included
 
    
    def actionSpaceSample(self):
        return np.random.choice(self.possibleActions)
        
    def posWithoutWalls(self, state):
        ...
        
    def posWithWalls(self, state):
        ...
        
    @property
    def maze(self):
        return self.__grid
    
    @property
    def robot(self):
        return self.__robot
    
    @property
    def entrance(self):
        return self.__start
    
    @property
    def goal(self):
        return self.__goal
    
    @property
    def game_over(self):
        return self.__game_over
    
    @property
    def MAZE_W(self):
        return int(self.maze_size[0])

    @property
    def MAZE_H(self):
        return int(self.maze_size[1])

    @property
    def WALLMAZE_W(self):
        return int(2*self.maze_size[0]-1)

    @property
    def WALLMAZE_H(self):
        return int(2*self.maze_size[1]-1)


    @property
    def SCREEN_SIZE(self):
        return tuple(self.__screen_size)
    
    @property
    def SCREEN_W(self):
        return int(self.SCREEN_SIZE[0])
    
    @property
    def SCREEN_H(self):
        return int(self.SCREEN_SIZE[1])
    
    @property
    def CELL_W(self):
        return float(self.SCREEN_W) / float(self.maze.MAZE_W)
    
    @property
    def CELL_H(self):
        return float(self.SCREEN_H) / float(self.maze.MAZE_H)
       
    def render(self, mode='human'):
        ...
    
    def close (self):
        ...
        
def maxAction(Q, state, actions):
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)
    return actions[action]        


if __name__ == "__main__":

    env = gridGeneratorEnv(maze_size = (4, 4))
    # model hyperparameters
    ALPHA = 0.1
    ALPHA_WALLS = 0.1
    GAMMA = 1.0
    GAMMA_WALLS = 1.0
    EPS = 1.0
    EPS_WALLS = 1.0

    
    Q = {}
    Q_walls = {}
    for state in env.stateSpace:
        for action in env.possibleActions:
            Q[state, action] = 0
    for wallstate in env.stateSpaceWalls:
        for wallaction in env.possibleActions:
            Q_walls[wallstate, wallaction] = 0

    numGames = 1000
    totalRewards = np.zeros(numGames)
    totalWallRewards = np.zeros(numGames)
    
    for i in range(numGames):
        if i % 5 == 0:
            print('starting game ', i)
        done = False
        epRewards = 0
        epWallRewards = 0 
        observation, wallobservation = env.reset()
    
        while not done:
#            print('hi')
            rand = np.random.random()
            rand_walls = np.random.random()

            action = maxAction(Q,observation, env.possibleActions) if rand < (1-EPS) \
                                                    else env.actionSpaceSample()
            wallaction = maxAction(Q_walls, wallobservation, env.possibleActions) if rand_walls < (1-EPS_WALLS) \
                                                    else env.actionSpaceSample()                                                 
                                                    
            observation_, wallobservation_, reward, reward_walls, done, info = env.step(action, wallaction)
            print(wallobservation_)
            epRewards += reward
            epWallRewards += reward_walls

            action_ = maxAction(Q, observation_, env.possibleActions)
            wallaction_ = maxAction(Q_walls, observation_, env.possibleActions)
            
            Q[observation, action] = Q[observation, action] + ALPHA*(reward + \
                        GAMMA*Q[observation_, action_] - Q[observation, action])

            Q_walls[wallobservation, wallaction] = Q_walls[wallobservation, wallaction] + ALPHA_WALLS*(reward_walls + \
                        GAMMA_WALLS*Q_walls[wallobservation_, wallaction_] - Q_walls[wallobservation, wallaction])

            observation = observation_
            wallobservation = wallobservation_
#        print(epWallRewards)
        if EPS - 2 / numGames > 0:
            EPS -= 2 / numGames
        else:
            EPS = 0
            
        if EPS_WALLS - 2 / numGames > 0:
            EPS_WALLS -= 2 / numGames
        else:
            EPS_WALLS = 0    
        totalRewards[i] = epRewards
        totalWallRewards[i] = epWallRewards
    
    plt.plot(totalRewards)
    plt.show()