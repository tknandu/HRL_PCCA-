import random
import sys
from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from rlglue.types import Observation
from rlglue.types import Action
from rlglue.types import Reward_observation_terminal

# This is a very simple discrete-state, episodic grid world that has 
# exploding mines in it.  If the agent steps on a mine, the episode
# ends with a large negative reward.
# 
# The reward per step is -1, with +10 for exiting the game successfully
# and -100 for stepping on a mine.


# TO USE THIS Environment [order doesn't matter]
# NOTE: I'm assuming the Python codec is installed an is in your Python path
#   -  Start the rl_glue executable socket server on your computer
#   -  Run the SampleSarsaAgent and SampleExperiment from this or a
#   different codec (Matlab, Python, Java, C, Lisp should all be fine)
#   -  Start this environment like:
#   $> python sample_mines_environment.py

class mines_environment(Environment):
	
	wind = True

	FREE = 0
	BAD = 1
	WORSE = 2
	WORST = 3
	GOAL = 4
	randGenerator = random.Random()
	fixedStartState=False;
	startRow = [5,6,10,11]
	startCol = 0

	def env_init(self):
		env_file = open(sys.argv[1],"r")
		self.map = []
	    	for l in env_file.readlines():
			self.map += [[int(i) for i in l.strip().split(',')]]

		#The Python task spec parser is not yet able to build task specs programmatically
		return "VERSION RL-Glue-3.0 PROBLEMTYPE episodic DISCOUNTFACTOR 0.9 OBSERVATIONS INTS (0 143) ACTIONS INTS (0 3) REWARDS (-3.0 10.0) EXTRA SampleMinesEnvironment(C/C++) by Brian Tanner."
	
	def env_start(self):
		if self.fixedStartState == False:
			stateValid=self.setAgentState(random.choice(self.startRow),self.startCol)
		else:
			stateValid=self.setAgentState(self.startRow,self.startCol)
		returnObs=Observation()
		returnObs.intArray=[self.calculateFlatState()]
		return returnObs
		
	def env_step(self,thisAction):
		# Make sure the action is valid 
		assert len(thisAction.intArray)==1,"Expected 1 integer action."
		assert thisAction.intArray[0]>=0, "Expected action to be in [0,3]"
		assert thisAction.intArray[0]<4, "Expected action to be in [0,3]"
		
		self.updatePosition(thisAction.intArray[0])

		theObs=Observation()
		theObs.intArray=[self.calculateFlatState()]

		returnRO=Reward_observation_terminal()
		returnRO.r=self.calculateReward()
		returnRO.o=theObs
		returnRO.terminal=self.checkCurrentTerminal()

		return returnRO

	def env_cleanup(self):
		pass

	def env_message(self,inMessage):
		#	Message Description
	 	# 'set-random-start-state'
	 	#Action: Set flag to do random starting states (the default)
		if inMessage.startswith("set-random-start-state"):
			self.fixedStartState=False;
			return "Message understood.  Using random start state.";

		#	Message Description
		# 'set-start-state X Y'
		# Action: Set flag to do fixed starting states (row=X, col=Y)
		if inMessage.startswith("set-start-state"):
			splitString=inMessage.split(" ");
			self.startRow=int(splitString[1]);
			self.startCol=int(splitString[2]);
			self.fixedStartState=True;
			return "Message understood.  Using fixed start state.";

		#	Message Description
		#	'print-state'
		#	Action: Print the map and the current agent location
		if inMessage.startswith("print-state"):
			self.printState();
			return "Message understood.  Printed the state.";

		return "SamplesMinesEnvironment(Python) does not respond to that message.";

	def setAgentState(self,row, col):
		self.agentRow=row
		self.agentCol=col

		return self.checkValid(row,col) and not self.checkTerminal(row,col)

	def setRandomState(self):
		numRows=len(self.map)
		numCols=len(self.map[0])
		startRow=self.randGenerator.randint(0,numRows-1)
		startCol=self.randGenerator.randint(0,numCols-1)

		while not self.setAgentState(startRow,startCol):
			startRow=self.randGenerator.randint(0,numRows-1)
			startCol=self.randGenerator.randint(0,numCols-1)

	def checkValid(self,row, col):
		valid=False
		numRows=len(self.map)
		numCols=len(self.map[0])

		if(row < numRows and row >= 0 and col < numCols and col >= 0):
			valid=True
		return valid

	def checkTerminal(self,row,col):
		if (self.map[row][col] == self.GOAL):
			return True
		return False

	def checkCurrentTerminal(self):
		return self.checkTerminal(self.agentRow,self.agentCol)

	def calculateFlatState(self):
		numRows=len(self.map)
		return self.agentCol * numRows + self.agentRow



	def updatePosition(self, theAction):
		# When the move would result in hitting an obstacles, the agent simply doesn't move 

		newRow = self.agentRow;
		newCol = self.agentCol;

		stoch = random.random()
		if(stoch >= 0.1):
			if (theAction == 0):#move left
				newCol = self.agentCol - 1;

			if (theAction == 1): #move right
				newCol = self.agentCol + 1;

			if (theAction == 2):#move up
				newRow = self.agentRow - 1;

			if (theAction == 3):#move down
				newRow = self.agentRow + 1;
		else:
			possibleActions=  [i for i in [0,1,2,3] if i!=theAction]
			newAction = random.choice(possibleActions)
			if (newAction == 0):#move left
				newCol = self.agentCol - 1;

			if (newAction == 1): #move right
				newCol = self.agentCol + 1;

			if (newAction == 2):#move up
				newRow = self.agentRow - 1;

			if (newAction == 3):#move down
				newRow = self.agentRow + 1;
			

		#Check if new position is out of bounds or inside an obstacle 
		if(self.checkValid(newRow,newCol)):
			self.agentRow = newRow;
			self.agentCol = newCol;

		stoch = random.random()
		if(stoch >= 0.5):
			#Blow right
			newCol = self.agentCol + 1
			if(self.checkValid(self.agentRow,newCol)):
				self.agentCol = newCol;


	def calculateReward(self):
		if(self.map[self.agentRow][self.agentCol] == self.GOAL):
			return 10.0;
		if(self.map[self.agentRow][self.agentCol] == self.BAD):
			return -1.0;
		if(self.map[self.agentRow][self.agentCol] == self.WORSE):
			return -2.0
		if(self.map[self.agentRow][self.agentCol] == self.WORST):
			return -3.0
		return 0.0;
		
	def printState(self):
		numRows=len(self.map)
		numCols=len(self.map[0])
		print "Agent is at: "+str(self.agentRow)+","+str(self.agentCol)
		print "Columns:0-10                10-17"
		print "Col    ",
		for col in range(0,numCols):
			print col%10,
			
		for row in range(0,numRows):
			print
			print "Row: "+str(row)+" ",
			for col in range(0,numCols):
				if self.agentRow==row and self.agentCol==col:
					print "A",
				else:
					if self.map[row][col] == self.GOAL:
						print "G",
					if self.map[row][col] == self.BAD:
						print "X",
					if self.map[row][col] == self.WORSE:
						print "Y",
					if self.map[row][col] == self.WORST:
						print "Z",
					if self.map[row][col] == self.FREE:
						print " ",
		print
		

if __name__=="__main__":
	EnvironmentLoader.loadEnvironment(mines_environment())
