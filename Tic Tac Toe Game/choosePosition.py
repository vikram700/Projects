# This function is used for choosing the position for entering the mark
import checkSpace as checkSpace

def choose(board):
    position = 0

    while position not in range(1,10) and checkSpace.freeSpace(board, position):
        try:
            position = int(input("Enter the position in range 1 to 9: "))
        except:
            continue
        
    return position
    