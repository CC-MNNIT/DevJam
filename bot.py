import json
import matplotlib.pyplot as plt

class Bot(object):


    def __init__(self):
        self.state = int(input("Enter 1 to trian or 2 to test (Default value: train):"))
        self.gameNo = 0
        self.DumpAfter = 25
        self.discount = 0.8
        self.r = {0: 1, 1:10, 2:-100 , 3: -1000}
        self.lr = 0.4
        if self.state is 2:
            self.lr = 0
        self.loadQval()
        self.lastState = "-20_-240_4"
        self.lastAction = 0
        self.moves = []
        self.score = 0
        self.avgScore = 0
        self.allScore = []
        self.x = []

#load privous trained data.
    def loadQval(self):
        self.Qval = {}
        fil = open('Qval.json', 'r')
        self.Qval = json.load(fil)
        fil.close()


#Decide whether to jump or not
    def act(self, difX, difY, vel):
        if difX > 140:
            difX = int(difX) - (int(difX) % 70)
        state = str(int(difX) - int(difX)%10) + '_' + str(int(difY) - int(difY)%10) + '_' + str(vel)  #%10 to reduce complexicity
        self.moves.append([self.lastState,self.lastAction,state])
        self.lastState = state


        try:
            ifDontFlap = self.Qval[state][0]
            ifFlaped = self.Qval[state][1]
        except :
            self.Qval.update({state : [0,0]})

        # if privous reward for "DoNotFlap" is higher than "Flap" then "DoNotFlap" in given state
        if self.Qval[state][0] >= self.Qval[state][1]:
            self.lastAction = 0
            return 0
        else:
            self.lastAction = 1
            return 1


#main function where q values are updated
    def update_scores(self):
        revMoves = list(reversed(self.moves))
        if (int(revMoves[0][2].split('_')[1]) > 100):
            upperPipeCol = True
        else:
            upperPipeCol = False
        t=1
        for m in revMoves:
            state = m[0]
            act = m[1]
            res = m[2]

            if t == 1 or t==2:
                self.Qval[state][act] = (1- self.lr) * (self.Qval[state][act]) + (self.lr) * ( self.r[3] + (self.discount)*max(self.Qval[res]) )
            #if t ==5 or t==6:
            #    self.Qval[state][act] = (1- self.lr) * (self.Qval[state][act]) + (self.lr) * ( self.r[1] + (self.discount)*max(self.Qval[res]) )
            elif upperPipeCol and act :
                self.Qval[state][act] = (1- self.lr) * (self.Qval[state][act]) + (self.lr) * ( self.r[3] + (self.discount)*max(self.Qval[res]) )
                upperPipeCol = False
            else:
                self.Qval[state][act] = (1- self.lr) * (self.Qval[state][act]) + (self.lr) * ( self.r[1] + (self.discount)*max(self.Qval[res]) )
            if (int(state.split("_")[1]) > -100 and int(state.split("_")[0]) < 0):
                self.Qval[state][act] = (1- self.lr) * (self.Qval[state][act]) + (self.lr) * ( self.r[0] + (self.discount)*max(self.Qval[res]) )
            t += 1
        self.gameNo += 1 #increase game count
        self.dump_Qval() # Dump q values (if game count % DumpAfter == 0)
        self.moves = []

    def dump_Qval(self):

        if self.gameNo % self.DumpAfter == 0:
            fil = open('Qval.json', 'w')
            json.dump(self.Qval, fil)
            fil.close()
            print('Q-values updated on local file.')

    def highScores(self,score):
        if self.score < score:
            self.score = score
        self.allScore.append(score)
        self.x.append(self.gameNo)
        self.avgScore = (self.avgScore*(self.gameNo-1) + score ) / self.gameNo
        print("Last Score:", score)
        print("Avg Score:",self.avgScore)
        print("Highest Score:",self.score)   #to check progress of learned game
        print("Game Count:",self.gameNo)
        if (self.gameNo%500 == 0):
            plt.plot(self.x, self.allScore)
            # naming the x axis
            plt.xlabel('Itrations')
            # naming the y axis
            plt.ylabel('Scores')

            # giving a title to my graph
            plt.title('For lr :' + str(self.lr) + 'and discount : ' + str(self.discount))

            # function to show the plot
            plt.show()
