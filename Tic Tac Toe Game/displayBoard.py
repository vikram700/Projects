def decoration():
    print("-------------------")

def display(board):
    for i in range(0,9,3):
        decoration()
        j = i + 1
        k = i + 2
        print("| ",board[i]," | ",board[j]," | ",board[k]," |")
    decoration()

 