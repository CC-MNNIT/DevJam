Team name: DreamHack

Project Description:

flappy(No rotation).py: Modified version of flappy bird. Some features are removed to make game more easy like rotation and acceleration.

flappy(with rotation).py: Original version of game.

"Music is removed" : to add it again : assets -> audio -> (Copy files from "Original audio to audio folder"

Bot.py: main algorithm of machine learning is applied here. Name of algorithm is "qLearn"

Q Learn:
	
	Q(s,a) = (1-lr)Q(s,a) + lr(r + discount*max(Q(s+1,a)))

Q(s,a) : is a function of state and action.

s : is the state of game which is unique for ecah and every state of game. It is used to identify the game position. In this game state is defined as a combied string made of horizontal seperation between bird and nearest pipe + vertical seperation between bird and nearest pipe + velocity of bird. ("difX_difY_vel)

a : is action performed in that given state.Here we have only two action to perform: to jump or not to jump.

lr : is learning rate  (0 <= lr <= 1)

r : is reward. We sat positive reward for action correctly performed and negative reward for undesired action in given state. 

discount : it encourages to think about future reward. like if in given state we provide '0' reward, then to make sure that algo is going in right direction we add this. (0 <= discount <= 1)

This algo try to maximize reward for given state. So if next time game reaches a known state, it will select that action with maximum reward.

To know more about this go here. [https://en.wikipedia.org/wiki/Q-learning]

To download this original game :
git clone https://github.com/sourabhv/FlappyBirdClone.git


Requirements-
pygame 1.91
Github URL : www.github.com/KashyapNasit
In case of trouble accessing the project use google drive link :
https://drive.google.com/drive/folders/1Tvt5p6G-QSQNfCM6nf9OufwJA_BPnhsc
