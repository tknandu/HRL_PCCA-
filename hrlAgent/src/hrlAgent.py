import random
import time
import string
import pickle
from rlglue.agent.Agent import Agent
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.agent import AgentLoader as AgentLoader
import numpy as np
from qnn import QNN
from pcca import PCCA
from deep_qnn import Deep_QNN
from tnn import TNN
import math

class Monster:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.sx = 0
        self.sy = 0
        self.type = -1
        self.typeName = ""
        self.winged = False

monsterNames = ["Mario", "Red Koopa", "Green Koopa", "Goomba", "Spikey", "Piranha Plant", "Mushroom", "Fire Flower", "Fireball", "Shell", "Big Mario", "Fiery Mario"]

'''
* Valid tiles:
* M - the tile mario is currently on. there is no tile for a monster.
* $ - a coin
* b - a smashable brick
* ? - a question block
* | - a pipe. gets its own tile because often there are pirahna plants
*     in them
* ! - the finish line
* And an integer in [1,7] is a 3 bit binary flag
*  first bit is "cannot go through this tile from above"
*  second bit is "cannot go through this tile from below"
*  third bit is "cannot go through this tile from either side"
'''
#This is the encoder for the single layer state
tileEncoder = {'\0': -1, '1': -1, '2': -1, '3': -1, '4': -1, '5': -1, '6': -1, '7': -1, 'M': 0, '$': 2, 'b': 2, '?': 2, '|': 2, '!': 2, ' ': 1, '\n': 1}

#These are the encoders for the multiple substrates state
backgroundLayer = {'\0': 2, '1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 2, 'M': 0, '$': 0, 'b': 0, '?': 0, '|': 0, '!': 0, ' ': 0, '\n': 0}
rewardLayer = {'\0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, 'M': 0, '$': 1, 'b': 1, '?': 1, '|': 1, '!': 2, ' ': 0, '\n': 0}

class MarioAgent(Agent):
    
    def agent_init(self,taskSpecString):
        self.policy_frozen = False
        self.total_steps = 0
        self.trial_start = 0.0
        self.total_steps = 0
        self.step_number = 0
        self.exp = 0.75 #Exploration factor
        self.substrate_mode = False
        if (self.substrate_mode):
            self.state_dim_x = 33
            self.state_dim_y = 7
        else:
            self.state_dim_x = 20
            self.state_dim_y = 12

        self.last_state = None
        self.last_action = None
        random.seed(0)
        
        #self.Q = QNN(nactions=12, input_size=(self.state_dim_x*self.state_dim_y), max_experiences=500, gamma=0.6, alpha=0.2)
        self.Q = Deep_QNN(nactions=12, input_size=(self.state_dim_x*self.state_dim_y), max_experiences=500, gamma=0.6, alpha=0.2)
        self.T = TNN(input_size= 4, max_experiences=500, alpha=0.2)

        #self.clusterer = PCCA()

        self.discretization_done = False
        
        self.n_bins = 10
        self.n_disc_states = (self.n_bins + 3)*(self.n_bins + 3)
        self.phi_mat = np.zeros((self.n_disc_states,12,self.n_disc_states),dtype=int)
        self.U_mat = np.zeros((self.n_disc_states,12,self.n_disc_states),dtype=float)

        self.regularization_constant = 0.4 #For rewards incorporated into transition structure

        #The (probably) incorrect U_mat that Peeyush writes in his algo
        self.Peey_U_mat = np.zeros((self.n_disc_states,12,self.n_disc_states),dtype=float)

#        self.transition_matrix = {} # this is actually only counts
#        self.transition_probs = {} # actual probabilities
        self.last_enc_state = None
        self.last_enc_action = None

        self.seen_expanded_states = {} #A dictionary of all the non-approximated (original Mario input) states that were seen.

    def agent_start(self,observation):
        self.step_number = 0
        self.trial_start = time.clock()

        # At the start of each episode, run PCCA
        #   chi_matrix = self.clusterer()
        self.seen_expanded_states[tuple(self.stateEncoder(observation))] = True

        act = self.getAction(observation)
        if self.discretization_done:
            enc_state = tuple(self.Q.getHiddenLayerRepresentation(self.stateEncoder(observation)))
            self.last_disc_state = self.getDiscretizedState(enc_state)
            self.last_enc_action = self.actionEncoder(act)
        return act
    
    def agent_step(self,reward, observation):
        #self.printFullState(observation)
        #self.printMarioState(observation)
        self.step_number += 1
        self.total_steps += 1
        act = self.getAction(observation)

        if (not self.transition_learning_frozen):
            enc_state = tuple(self.Q.getHiddenLayerRepresentation(self.stateEncoder(observation)))
            disc_state = self.getDiscretizedState(enc_state)
            enc_action = self.actionEncoder(act)        
            self.phi_mat[self.last_disc_state][self.last_enc_action][disc_state] += 1
            self.U_mat[self.last_disc_state][self.last_enc_action][disc_state] += math.exp(-self.regularization_constant*abs(reward))
            self.Peey_U_mat[self.last_disc_state][self.last_enc_action][disc_state] += self.phi_mat[self.last_disc_state][self.last_enc_action][disc_state]*math.exp(-self.regularization_constant*abs(reward))
            """
            if disc_state not in self.:
                self.transition_matrix[disc_state] = {}
            if self.disc_enc_action:
                if self.last_enc_action not in self.transition_matrix[self.last_enc_state]:
                    self.transition_matrix[self.last_enc_state][self.last_enc_action] = {enc_state:1}
                elif enc_state not in self.transition_matrix[self.last_enc_state][self.last_enc_action]:
                    self.transition_matrix[self.last_enc_state][self.last_enc_action][enc_state] = 1
                else:
                    self.transition_matrix[self.last_enc_state][self.last_enc_action][enc_state] += 1
            """
            self.last_enc_state = enc_state
            self.last_enc_action = enc_action
            self.last_disc_state = disc_state

        if (not self.policy_frozen):
            self.seen_expanded_states[tuple(self.stateEncoder(observation))] = True
            self.update(observation, act, reward)

        self.last_state = observation
        self.last_action = act
        return act
    
    def agent_end(self,reward):
        time_passed = time.clock() - self.trial_start
        print "ended after " + str(self.total_steps) + " total steps"
        print "average " + str(self.step_number/time_passed) + " steps per second"
    
    def agent_cleanup(self):
        pass
    
    def agent_freeze(self):
        pass
    
    def agent_message(self,inMessage):
        if inMessage.startswith("freeze_learning"):
            self.policy_frozen=True
            return "message understood, policy frozen"
        if inMessage.startswith("unfreeze_learning"):
            self.policy_frozen=False
            return "message understood, policy unfrozen"
        if inMessage.startswith("set_exploring"):
            splitString=inMessage.split(" ")
            self.exp = float(splitString[1])
            return "message understood, setting exploration factor"
        if inMessage.startswith("save_policy"):
            splitString=inMessage.split(" ")
            self.saveQFun(splitString[1])
            print "Saved."
            return "message understood, saving policy"
        if inMessage.startswith("load_policy"):
            splitString=inMessage.split(" ")
            self.loadQFun(splitString[1])
            print "Loaded."
            return "message understood, loading policy"
        if inMessage.startswith("use_impactful_experiences"):
            self.Q.use_impactful = True
            return "message understood, using impactful experiences"
        if inMessage.startswith("use_all_experiences"):
            self.Q.use_impactful = False
            return "message understood, using all experiences"
        if inMessage.startswith("reset_q"):
            self.Q = QNN(nactions=12, input_size=(self.state_dim_x*self.state_dim_y), max_experiences=500, gamma=0.6, alpha=0.2)
            return "message understood, reseting q-function"

        # Added now
        if inMessage.startswith("freeze_transition_learning"):
            self.transition_learning_frozen=True
            return "message understood, transition learning frozen"    
        if inMessage.startswith("unfreeze_transition_learning"):
            self.transition_learning_frozen=False
            return "message understood, transition learning unfrozen"        
        if inMessage.startswith("train_TNN"):
            print 'Training TNN'
            self.trainTNN()
        if inMessage.startswith("savetransmatrix"):
            splitString=inMessage.split(" ")
            self.saveTprobs(splitString[1])
            print "Saved.";
            return "message understood, saving Tprobs"
        if inMessage.startswith("loadtransmatrix"):
            splitString=inMessage.split(" ")
            self.loadTprobs(splitString[1])
            print "Loaded.";
            return "message understood, loading Tprobs"
        if inMessage.startswith("save_state_reps"): #Save the tuples corresponding to each state.
            enc_states = []
            for state in self.seen_expanded_states.keys():
                enc_states.append(tuple(self.Q.getHiddenLayerRepresentation(list(state))))
            splitstring = inMessage.split()
            outfile = open(splitstring[1],'w')
            pickle.dump(enc_states,outfile)
        #Once we have frozen our state reps, this function discretizes them and populates the self.secondColBins and self.thirdColBins arrays
        if inMessage.startswith("get_bins_from_state_reps"):
            enc_states = []
            for state in self.seen_expanded_states.keys():
                curr_rep = tuple(self.Q.getHiddenLayerRepresentation(list(state)))
                enc_states.append([curr_rep[0][0], curr_rep[1][0], curr_rep[2][0]])
            enc_states = np.array(enc_states)
            second_col = enc_states[:,1]
            self.secondColBins = self.getBins(second_col)
            third_col = enc_states[:,2]
            self.thirdColBins = self.getBins(third_col)

            self.discretization_done = True

        if inMessage.startswith("save_phi_u_peeyu"):
            splitstring = inMessage.split()
            phioutfile = open(splitstring[1],'w')
            pickle.dump(self.phi_mat,phioutfile)

            uoutfile = open(splitstring[2],'w')
            pickle.dump(self.U_mat,uoutfile)

            puoutfile = open(splitstring[3],'w')
            pickle.dump(self.Peey_U_mat,puoutfile)
        return None

    #Get the discretized states. Assumes that the bins are available.
    def getDiscretizedState(self,enc_state):
        
        second_elem = enc_state[1][0]
        third_elem = enc_state[2][0]
        disc_tuple = (np.digitize([second_elem],self.secondColBins)[0],np.digitize([third_elem],self.thirdColBins)[0])
        return (self.n_bins+3)*disc_tuple[0]+disc_tuple[1]

    def getBins(self,i_array):
        bins = [0.0]
        i_array.sort()

        i = 0
        while i < len(i_array):
            bins.append(i_array[i])
            i += len(i_array)/self.n_bins
        bins.append(1.0)
        return np.array(bins)
                
    def getMonsters(self, observation):
        monsters = []
        i = 0
        while (1+2*i < len(observation.intArray)):
            m = Monster()
            m.type = observation.intArray[1+2*i]
            m.winged = (observation.intArray[2+2*i]) != 0
            m.x = observation.doubleArray[4*i];
            m.y = observation.doubleArray[4*i+1];
            m.sx = observation.doubleArray[4*i+2];
            m.sy = observation.doubleArray[4*i+3];
            m.typeName = monsterNames[m.type]
            monsters.append(m)
            i += 1
        return monsters

    def getMario(self, observation):
        monsters = self.getMonsters(observation)
        for i in xrange(len(monsters)):
            if (monsters[i].type == 0 or monsters[i].type == 10 or monsters[i].type == 11):
                return (monsters[i].x-observation.intArray[0], monsters[i].y)
        return None

    def getMarioFromTiles(self, observation):
        for xi in xrange(22):
            for yi in xrange(16):
                if (self.getTileAt(xi, yi, observation) == 'M'):
                    return (xi, yi)
        return None

    def getTileAt(self, x, y, observation):
        if (x < 0):
            return '7'
        y = 16 - y
        x -= observation.intArray[0]
        if (x<0 or x>21 or y<0 or y>15):
            return '\0';
        index = y*22+x;
        return observation.charArray[index];        

    #Handy little function for debugging by printing out the char array
    def printFullState(self, observation):
        print "-----------------"
        for yi in xrange(16):
            out_string = ""
            for xi in xrange(22):
                out_string += observation.charArray[yi*22+xi]
            print out_string

    #Handy little functio for debugging by printing out Mario's (x,y) position
    def printMarioState(self, observation):
        mar = self.getMario(observation)
        print "Mario X: " + str(mar[0]) + " Mario Y: " + str(mar[1])

    #Handy little function for debugging by printing out the encoded state that is being send to the NN
    def printEncodedState(self, s):
        print "-----------------"
        for yi in xrange(self.state_dim_y):
            out_string = ""
            for xi in xrange(self.state_dim_x):
                out_string += (str(s[yi*self.state_dim_x+xi]) + " ")
            print out_string

    def getAction(self, observation):
        act = self.qnnAction(observation)
        return act

    def update(self, observation, action, reward):
        self.qnnUpdate(observation, action, reward)
        self.Q.ExperienceReplay()

    def stateEncoder(self, observation):
        if (not self.substrate_mode):
            return self.stateEncoderSingle(observation)
        else:
            return self.stateEncoderMultiple(observation)
        pass

    def stateEncoderSingle(self, observation):
        s = []
        #Determine Mario's current position. Everything is relative to Mario
        mar = self.getMario(observation)
        mx = int(mar[0])
        my = 15 - int(mar[1])
        #Update based on the environment
        for yi in xrange(self.state_dim_y):
            for xi in xrange(self.state_dim_x):
                x = mx + xi - int(self.state_dim_x/2.0)
                y = my + yi - int(self.state_dim_y/2.0)
                if (x < 0 or x > 21 or y < 0 or y > 15):
                    s.append(-1)
                    continue
                s.append(tileEncoder[observation.charArray[y*22+x]]) # shouldn't this be 20
        #Add monsters
        monsters = self.getMonsters(observation)
        for mi in xrange(len(monsters)):
            if (monsters[mi].type == 0 or monsters[mi].type == 10 or monsters[mi].type == 11): #skip mario
                continue
            monx = int(monsters[mi].x)
            mony = 15 - int(monsters[mi].y)
            x = monx - mx + int(self.state_dim_x/2.0) - observation.intArray[0]
            y = mony - my + int(self.state_dim_y/2.0)
            if (x < 0 or x >= self.state_dim_x or y < 0 or y >= self.state_dim_y): #skip monsters farther away
                continue
            s[y*self.state_dim_x + x] = -2
        return s

    def stateEncoderMultiple(self, observation):
        s1 = []
        s2 = []
        s3 = []
        #Determine Mario's current position. Everything is relative to Mario
        mar = self.getMario(observation)
        mx = int(mar[0])
        my = 15 - int(mar[1])
        #Add background and reward layers with a placeholder monster layer
        for yi in xrange(self.state_dim_y):
            for xi in xrange(self.state_dim_x/3):
                x = mx + xi - int(self.state_dim_x/6.0)
                y = my + yi - int(self.state_dim_y/2.0)
                if (x < 0 or x > 21 or y < 0 or y > 15):
                    s1.append(0)
                    s2.append(0)
                    s3.append(0)
                    continue
                s1.append(backgroundLayer[observation.charArray[y*22+x]])
                s2.append(rewardLayer[observation.charArray[y*22+x]])
                s3.append(0)
        #Add monsters to the monster layer
        monsters = self.getMonsters(observation)
        for mi in xrange(len(monsters)):
            if (monsters[mi].type == 0 or monsters[mi].type == 10 or monsters[mi].type == 11):#skip mario
                continue
            monx = int(monsters[mi].x)
            mony = 15 - int(monsters[mi].y)
            x = monx - mx + int(self.state_dim_x/6.0) - observation.intArray[0]
            y = mony - my + int(self.state_dim_y/2.0)
            if (x < 0 or x >= self.state_dim_x/3 or y < 0 or y >= self.state_dim_y): #skip monsters farther away
                continue
            s3[y*self.state_dim_x/3 + x] = 1
        return s1 + s2 + s3

    def actionEncoder(self, act):
        a = 1*act.intArray[2] + 2*act.intArray[1] + 4*(act.intArray[0]+1)
        return a

    def actionDecoder(self, a):
        act = Action()
        act.intArray.append(int(a/4.0) - 1)
        act.intArray.append(int(a/2.0) % 2)
        act.intArray.append(a % 2)
        return act

    def randomAction(self, forwardBias):
        act = Action()
        #The first control input is -1 for left, 0 for nothing, 1 for right
        if (not forwardBias or random.random() < 0.1):
            act.intArray.append(random.randint(-1,1))
        else:
            act.intArray.append(random.randint(0,1))
        #The second control input is 0 for nothing, 1 for jump
        act.intArray.append(random.randint(0,1))
        #The third control input is 0 for nothing, 1 for speed increase
        act.intArray.append(random.randint(0,1))        
        return act

    def qnnAction(self, observation):
        s = self.stateEncoder(observation)
        #self.printEncodedState(s)
        if (random.random()>self.exp):
            a = np.argmax(self.Q(s))
            act = self.actionDecoder(a)
        else:
            act = self.randomAction(True)
        return act

    def qnnUpdate(self, observation, action, reward):
        if (self.last_state == None or self.last_action == None):
            return
        s = self.stateEncoder(observation)
        ls = self.stateEncoder(self.last_state)
        a = self.actionEncoder(action)
        la = self.actionEncoder(self.last_action)
        self.Q.RememberExperience(ls, la, reward, s, a)

    def saveQFun(self, fileName):
        theFile = open(fileName, "w")
        pickle.dump(self.Q, theFile)
        theFile.close()

    def loadQFun(self, fileName):
        theFile = open(fileName, "r")
        self.Q = pickle.load(theFile)
        theFile.close()

    # Added now
    def train_TNN(self):
        for s1 in self.transition_probs:
            for a in self.transition_probs[s1]:
                for s2 in self.transition_probs[s1][a]:
                    self.TNN.Update(np.asarray(s1),np.asarray(s2),self.transition_probs[s1][a][s2])

    # TODO - take the current transition_matrix and compute transition_probs (&save it)
    def saveTprobs(self, fileName):
        transition_probs = {}
        for s1 in self.transition_matrix.keys():
            transition_probs[s1] = {}
            for a in self.transition_matrix[s1].keys():
                transition_probs[s1][a] = {}
                Z = 0 #Normalisation constant
                for s2 in self.transition_matrix[s1][a].keys():
                    Z += self.transition_matrix[s1][a][s2]
                for s2 in self.transition_matrix[s1][a].keys():
                    transition_probs[s1][a][s2] = float(self.transition_matrix[s1][a][s2])/float(Z)
        theFile = open(fileName,'w')
        pickle.dump(transition_probs,theFile)

    def loadTprobs(self, fileName):
        theFile = open(fileName, "r")
        self.transition_probs = pickle.load(theFile)
        theFile.close()

if __name__=="__main__":        
    AgentLoader.loadAgent(MarioAgent())
