def freeSpace(board, position):
    return board[position - 1] == ' '

def fullSpace(board):
    for i in range(1,10):
        if(freeSpace(board, i)):
            return False
    return True
