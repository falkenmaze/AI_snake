import torch
import random
import numpy as np
from collections import deque 
from game import SnakeGame, Direction, Point 
from neural_network import QNetwork, QTrainer

MAX_MEM = 100_000
BATCH_SIZE = 1000 
LEARNING_RATE = 0.01 

class Agent:
	def __init__(self):
		self.no_games = 0 
		self.randomness = 0
		self.gamma = 0.9 
		self.memory = deque(maxlen=MAX_MEM)
		self.model = QNetwork(11,256,3)
		self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)

	def get_state(self,game):
		head = game.snake[0]
		point_l = Point(head.x - 20, head.y)
		point_r = Point(head.x + 20, head.y)
		point_u = Point(head.x,head.y- 20)
		point_d = Point(head.x, head.y+ 20)

		dir_l = game.direction == Direction.LEFT 
		dir_r = game.direction == Direction.RIGHT 
		dir_u = game.direction == Direction.UP 
		dir_d = game.direction == Direction.DOWN 

		state = [
		#straight danger
		(dir_r and game.collide(point_r)) or
		(dir_l and game.collide(point_l)) or 
		(dir_u and game.collide(point_u)) or 
		(dir_d and game.collide(point_d)),

		#right danger
		(dir_u and game.collide(point_r)) or 
		(dir_d and game.collide(point_l)) or 
		(dir_l and game.collide(point_u)) or 
		(dir_r and game.collide(point_d)),

		#left danger
		(dir_d and game.collide(point_r)) or 
		(dir_u and game.collide(point_l)) or 
		(dir_r and game.collide(point_u)) or 
		(dir_l and game.collide(point_d)),

		#Move dir
		dir_l,
		dir_r,
		dir_u,
		dir_d,
		game.food.x < game.head.x,
		game.food.x > game.head.x, 
		game.food.y < game.head.y, 
		game.food.y > game.head.y 
		]
		return np.array(state, dtype=int)

	def remember(self, state, action, reward, next_state, game_over):
		self.memory.append((state,action,reward,next_state,game_over))

	def train_long_mem(self):
		if len(self.memory) > BATCH_SIZE:
			sample = random.sample(self.memory, BATCH_SIZE)
		else:
			sample = self.memory 
		states,actions,rewards,next_states,game_overs = zip(*sample)
		self.trainer.train_step(states,actions,rewards,next_states,game_overs)

	def train_short_mem(self, state,action,reward,next_state,game_over):
		self.trainer.train_step(state,action,reward,next_state,game_over)

	def get_action(self, state):
		# self.randomness = 70 - self.no_games
		# moves = [0,0,0]
		# #tradeoff exploration - exploitation
		# if random.randint(0,200) < self.randomness:
		# 	move = random.randint(0,2)
		# 	moves[move] = 1 
		# else:
		moves = [0,0,0]
		state0 = torch.tensor(state, dtype=torch.float)
		prediction = self.model(state0)
		move = torch.argmax(prediction).item()
		moves[move] = 1 
		return moves 

def train():
	record = 0 
	agent = Agent()
	game = SnakeGame()
	while True:
		state0 = agent.get_state(game)
		final_move = agent.get_action(state0)
		reward,game_over,score = game.play_step(final_move)
		state1 = agent.get_state(game)
		agent.train_short_mem(state0,final_move,reward,state1,game_over)
		agent.remember(state0, final_move,reward,state1,game_over)
		if game_over:
			game.reset()
			agent.no_games += 1
			agent.train_long_mem()

			if score > record:
				record = score 
				agent.model.save()

			print("Game ", agent.no_games, "Score ", score, 'Record ', record)

if __name__ == '__main__':
	train()
