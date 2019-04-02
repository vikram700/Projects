import random

def choose():
    if(random.randint(0,1) == 0):
        return 'player1'
    else:
        return 'player2'