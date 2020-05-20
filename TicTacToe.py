#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:06:40 2020

@author: mac
"""
import numpy as np
import random
import pandas as pd
from progressbar import ProgressBar
pbar = ProgressBar()

#tasks to do:
#display board()
#Choose move
#play move
#Choose symbols
memory = pd.read_csv('memory.csv')
memory = memory.set_index('state')
if 'Unnamed: 0' in memory.columns:
    memory = memory.drop('Unnamed: 0', axis = 1)
#state = '_+_+_+_+_+_+ + + '
#memory = pd.DataFrame(
 #   {'TL' : [0], 'TC' : [0], 'TR' : [0],
  #  'ML' : [0], 'MC' : [0], 'MR' : [0],
 #   'BL' : [0], 'BC' : [0], 'BR' : [0]
 #   },
 #   index = [state]
#)
initial_state = '_+_+_+_+_+_+ + + '
exploration_rate = .3
def play_game(memory):
    """
    Initializes the game
    """
    #store board as dictionary:
    board_state = {'TL':'_','TC':'_','TR':'_',
                   'ML':'_','MC':'_','MR':'_',
                   'BL':' ','BC':' ','BR':' ',}
    #Move names:
    move_dictionary = {'TL':'Top Left','TC':'Top Center','TR':'Top Right',
                   'ML':'Middle Left','MC':'Middle Center','MR':'Middle Right',
                   'BL':'Bottom Left','BC':'Bottom Center','BR':'Bottom Right'}
        
    def printboard(board_state):
        """
        Board_state is a dictionary containing where moves have been played
        Prints board with symbols
        """
        print('Tic Tac Toe')
        print(f'_{board_state["TL"]}|_{board_state["TC"]}|_{board_state["TR"]}')
        print(f'_{board_state["ML"]}|_{board_state["MC"]}|_{board_state["MR"]}')
        print(f' {board_state["BL"]}| {board_state["BC"]}| {board_state["BR"]}')
    def get_side():
        """
        Asks user to choose x or o, returns sides for each player
        """
        print('Which side do you want to be? (X or O)')
        side = input().upper()
        if not side in ['X', 'O']:
            print("Pick X or O:")
            get_side()
        elif side =='O':
            other_side = 'X'
        else: other_side = 'O'
        return(side, other_side)
        
    
    def get_possible_moves(board_state):
        """
        Takes board_state dictionary
        returns list of possible moves
        """
        possible_moves = []
        for key in board_state.keys():
            
            if board_state[key] in {'_', ' '}:
                possible_moves.append(key)
        return(possible_moves)
    
    def print_moves(possible_moves, move_dictionary):
        """
        Possible moves is list, movedictionary is dicitonary of move names
        
        prints out moves and abbreviations
        """
        move_dictionary = {'TL':'Top Left','TC':'Top Center','TR':'Top Right',
                   'ML':'Middle Left','MC':'Middle Center','MR':'Middle Right',
                   'BL':'Bottom Left','BC':'Bottom Center','BR':'Bottom Right'}
        print('Possible move (abbreviations)')
        for key, value in move_dictionary.items():
            print(value, f" ({key})")
            
    def human_move(board_state, possible_moves, side):
        """
        Boardstate is dictionary of board at this point, possible moves is list of empty squares
        side is symbol for human player
        
        takes input for next move and updates board
        """
        print('Where would you like to move? (enter the abbreviation)')
        move = input().upper()
        if not move in possible_moves:
            print('Sorry, that is not a possible move.')
            human_move(board_state, possible_moves, side)
        else:
            board_state[move] = side
    def computer_move_random(possible_moves):
            """
            takes possible moves dictionary argument
            changes board to reflect random computer move
            returns none
            """
            move = random.choice(possible_moves)
            return(move)
    def game_state_value(board_state):
        """
        Parameters
        ----------
        board_state : dictionary.

        Returns
        1 if won, 0 if tie, -1 if loss
        -------
        """
        game_state = is_over(board_state)
        if game_state == other_side:
            return(1)
        elif game_state == side:
            return(-1)
        else:
            return(0)
        
    def maximize(board_state):
        """
        Takes board state dictionary
        recursively calls minimize
        returns tuple of the maximum of lower calls or result of game in base case
        """
        imagine = board_state.copy()
        if not is_over(imagine) == False:
            return(None,(game_state_value(imagine)))
        else:
            move_dict = {}
            
            for move in get_possible_moves(board_state):
                imagine = board_state.copy()
                imagine[move] =  other_side
                move_dict[move] = minimize(imagine)[1]
        return((max(move_dict, key=move_dict.get), max(move_dict.values())))
    
    def minimize(board_state):
        """
        Takes board state dictionary
        recursively calls maximiz
        returns tuple of the minimum of lower calls or result of game in base case
        """
        imagine = board_state.copy()
        if not is_over(imagine) == False:
            return(None, game_state_value(board_state))
        else:
            
            move_dict = {}
            
            for move in get_possible_moves(board_state):
                imagine = board_state.copy()
                imagine[move] = side
                move_dict[move] = maximize(imagine)[1]
        return((min(move_dict, key=move_dict.get) , min(move_dict.values())))

    def computer_move_hard(possible_moves, board_state):
        """
        Parameters
        ----------
        possible_moves : Tlist of possible moves
        board_state : dictionary of board state

        Returns
        -------
        best move.

        """
        best_move = ''
        move_score = -2
        move, score = maximize(board_state)
        if move_score < int(score):
            best_move = move
            move_score = score
        
        return(best_move)

        
    class QLearning_agent(object):
        def __init__(self, memory, exploration_rate, side):
            self.side = 'O'
            self.states = []
            self.states_value = memory
            self.learning_rate = 0.2
            self. exploration_rate = exploration_rate
            self.decay_gamma = .9
            self.moves = []
        def choose_action(self, possible_moves, board_state):
            """
            Possible moves is list of open positions, board_state is location-value dictionary
            of open, X or O, self.side is X or O

            Returns a move as a string
            """
            decision = ['',-1]#store move values in tuple
            if np.random.uniform(0,1) <= self.exploration_rate:
                #take random action
                decision[0] = random.choice(possible_moves)
            else:#look at memory of move values and get best move
                state = self.get_hashable_state(board_state)
                if not state in self.states_value.index:
                    self.states_value.loc[state] = {'TL':0, 'TC':0, 'TR':0, 'ML':0, 'MC':0, 'MR':0, 'BL':0, 'BC':0, 'BR':0}
                for action in possible_moves:
                    if self.states_value.loc[state, action] >= decision[1]:
                        decision = (action, self.states_value.loc[state, action])
            return(decision[0])
        def remember_actions(self, board_state, move):
            """
            takes in hashable board state string and move string
            appends board state string to states list
            appends move to moves list
            returns none
            """
            
            self.states.append(self.get_hashable_state((board_state)))
            self.moves.append(move)
        def get_hashable_state(self, board_state):
            """
            takes in boardstate dictionary, returns hashable string
            """
            state = []
            for spot in ['TL', 'TC', 'TR', 'ML', 'MC', 'MR', 'BL', 'BC', 'BR']:
                state.append(board_state[spot])
            state = '+'.join(state)
            return(state)
        def board_from_hash(self, hashable_state):
            """
            takes in a state string
            splits on white space and recombines into state dictionary
            returns dictionary
            """
            spots = ['TL', 'TC', 'TR', 'ML', 'MC', 'MR', 'BL', 'BC', 'BR']
            board_state = {}
            for index, state in enumerate(hashable_state.split('+')):
                board_state[spots[index]] = state
            return(board_state)
        def get_reward(self, winner):
            """
            returns appropriate reward as int from winner = is_over()
            """
            
            if winner == 'O':
                reward = 1
            elif winner == 'X':
                reward = 0
            else:
                reward = 0.5
            return(reward)
        def maxaQ(self, state):
            """
            takes in state string, finds the best move Q value
            returns the highest Q value
            """
            
            if state in self.states_value.index:
                maxQ = max(self.states_value.loc[state])
            else:
                maxQ = 0
            return(maxQ)
        def learn(self, reward):
            """
            reward from get_reward() is int, from result of game. 1 for win, .5 for tie, 0 for lose
            updates Q values using states and actions from the most recent game
            Q algorithm for State S and action A, alpha = learning rate, gamma= discount:
            Q(S, A) := (1-alpha)*Q(S, A) + alpha*(reward + gamma*maxaQ(S',a))

            returns Qtable as a pd.dataframe
            """
            
            
            moves = self.moves
            states = self.states
            
            
            if len(moves) != len(states):
                print('moves and states have different lengths')
                
            for index in reversed(range(0, len(moves))):
                if not states[index] in self.states_value.index:
                    self.states_value.loc[states[index]] = {'TL':0, 'TC':0, 'TR':0, 'ML':0, 'MC':0, 'MR':0, 'BL':0, 'BC':0, 'BR':0}
                elif index == len(states)-1:
                    self.states_value.loc[states[index], moves[index]] = ((1-self.learning_rate)*self.states_value.loc[states[index], moves[index]]) + (self.learning_rate *reward)
                else:
                    MaxQ = self.maxaQ(states[index+1])
                    self.states_value.loc[states[index], moves[index]] = (1-self.learning_rate)*self.states_value.loc[states[index], moves[index]] + self.learning_rate*self.decay_gamma *(MaxQ)
            self.moves = []#erase memory of last game
            self.states = []
            return(self.states_value)
        def AItrain(self, iterations):
            """
            trains the Qtable for some integer number of iterations
            
            Returns
            -------
            trained Qtable as pandas dataframe.

            """
            memory = self.states_value
            import matplotlib.pyplot as plt
            import seaborn as sns
            (side, other_side) = ('X', 'O')
            outcomes = []
            pbar.currval = 0
            for iteration in pbar(range(1,iterations)):
                winner = False
                board_state = {'TL':'_','TC':'_','TR':'_',
                               'ML':'_','MC':'_','MR':'_',
                               'BL':' ','BC':' ','BR':' ',}
                
                while winner == False:
                    possible_moves = get_possible_moves(board_state)
                    board_state[computer_move(possible_moves, 'easy')] = 'X'
                    possible_moves = get_possible_moves(board_state)
                    winner = is_over(board_state)
                    if not winner == False:
                        break
                    move = self.choose_action(possible_moves, board_state)
                    self.remember_actions(board_state, move)
                    board_state[move] = 'O'
                    winner = is_over(board_state)
                reward = self.get_reward(winner)
                if winner == 'O':
                    outcomes.append(1)
                elif winner == 'X':
                    outcomes.append(0)
                else:
                    outcomes.append(.5)
                memory = self.learn(reward)
                self.moves = []
                self.states = []
            print("Training Complete")
            wins = np.mean(outcomes)
            
            print('Average performane:', wins)

            return(memory)
    def computer_move(possible_moves, mode):
        """
        hard is a string, either "easy", "medium", or "hard"
        returns computer move given easy, hard or medium mode
        hard mode is Maximin Algorithm
        easy is random
        medium is a random choice between the two.
        """
        if mode == 'easy':
            move = computer_move_random(possible_moves)
        elif mode == 'hard':
            move = computer_move_hard(possible_moves, board_state)
        else:#medium mode is combination of good and random moves
            move = random.choice([computer_move_random(possible_moves), computer_move_hard(possible_moves, board_state)])
        return(move)
    def is_over(board_state):
        """
        Takes board state dictionary
        
        Checks for three in a row. If it finds one, it returns the side that has three in a row 
        (returns the winning side). Then checks for full board and returns cat's game Otherwise, returns false
        """
        #Rows:
        if board_state['TL'] == board_state['TC'] == board_state['TR'] != '_':
            return(board_state['TL'])
        elif board_state['ML'] == board_state['MC'] == board_state['MR'] != '_':
            return(board_state['ML'])
        elif board_state['BL'] == board_state['BC'] == board_state['BR'] != ' ':
            return(board_state['BL'])
        #Columns:
        elif board_state['TL'] == board_state['ML'] == board_state['BL'] != '_':
            return(board_state['TL'])
        elif board_state['TC'] == board_state['MC'] == board_state['BC'] != '_':
            return(board_state['TC'])
        elif board_state['TR'] == board_state['MR'] == board_state['BR'] != '_':
            return(board_state['TR'])
        #Diagonals:
        elif board_state['BL'] == board_state['MC'] == board_state['TR'] != '_':
            return(board_state['BL'])
        elif board_state['TL'] == board_state['MC'] == board_state['BR'] != '_':
            return(board_state['TL'])
        elif get_possible_moves(board_state) == []:
            return("Cat's Game")
        else:
            return(False)
    def AImode(memory, agent):
        print('Play or Train?', end = '')
        mode = input().lower()
        if mode == 'play':
            side, other_side = get_side()#choose sides
            printboard(board_state)
            winner = False
            while winner == False:#play until winner or cat's game
                possible_moves = get_possible_moves(board_state)
                print_moves(possible_moves, move_dictionary)#show choices
                human_move(board_state, possible_moves, side)#make move
                printboard(board_state)
                
                
                winner = is_over(board_state)
                if not winner == False:
                    game_over(winner, side)
                    
                possible_moves = get_possible_moves(board_state)
                if possible_moves == []:
                    game_over("Cat's Game", side)
                #computer's turn:
                agent_move = agent.choose_action(possible_moves, board_state)
                agent.remember_actions(board_state, agent_move)
                board_state[agent_move] = other_side
                print('Computer plays:')
                printboard(board_state)
                
                winner = is_over(board_state)
    
            reward = agent.get_reward(winner) 
            agent.learn(reward)
            memory = agent.states_value
            memory.to_csv('memory.csv')
            game_over(winner, side)
        else:
            print('Number of iterations:', end = '')
            iterations = int(input())
            (side, other_side) = ('O', 'X')
            agent = QLearning_agent(memory, exploration_rate, 'O')
            memory = agent.AItrain(iterations)
            memory.to_csv('memory.csv')
            AImode(memory, agent)
    def game_over(winner, side):
        """
        Takes winner argument from is_over, and side. returns appropriate response and asks
        if user wants to play again.
        """
        if winner == side:
            print('Congratulations, You win!')
            print("Play again?(y/n)")
            again = input()
            if again.lower() in ['y', 'yes']:
                play_game(memory)
        elif winner == "Cat's Game":
            print("Cat's Game!")
            print("Play again?(y/n)")
            again = input()
            if again.lower() in ['y', 'yes']:
                play_game(memory)
        else:
            print("Better luck next time!")
            print("Play again?(y/n)")
            again = input()
            if again.lower() in ['y', 'yes']:
                play_game(memory)
                    
    print('Tic Tac Toe')
    print('Select Level: Easy, Medium, Hard or AI?')
    mode = input().lower()
    if not mode.lower() == 'ai':
        side, other_side = get_side()#choose sides
        printboard(board_state)
        winner = False
        while winner == False:#play until winner or cat's game
            possible_moves = get_possible_moves(board_state)
            print_moves(possible_moves, move_dictionary)#show choices
            human_move(board_state, possible_moves, side)#make move
            printboard(board_state)
            
            
            winner = is_over(board_state)
            if not winner == False:
                game_over(winner, side)
                
            possible_moves = get_possible_moves(board_state)
            if possible_moves == []:
                game_over("Cat's Game", side)
            board_state[computer_move(possible_moves, mode)] = other_side
            print('Computer plays:')
            printboard(board_state)
            
            winner = is_over(board_state)
            
        game_over(winner, side)
    else:
        AImode(memory, QLearning_agent(memory, exploration_rate, 'O'))
            
if __name__ == '__main__':
    play_game(memory)

