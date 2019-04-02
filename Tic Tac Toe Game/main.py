print("Welcome in Tic Toc Game")

board = [" "]*9
import inputMarker 
import displayBoard 
import chooseTurn 
import placeMarker 
import winCheck 
import checkSpace 
import replay 
import choosePosition

while True:

    board = [" "]*9
    displayBoard.display(board)
    player1_marker, player2_marker = inputMarker.takeInput()
    turn = chooseTurn.choose()

    print(turn + " will play first!")

    play_game = input("do you want to play game yes or no : ")

    if play_game :
        game_on = True
    else:
        game_on = False

    while game_on:
        if turn == 'player1':
           print("player first is playing !")
           displayBoard.display(board)
           position = choosePosition.choose(board)
           placeMarker.placeMark(board, player1_marker, position)

           if winCheck.check(board, player1_marker):
               print("Player_1 have won the match! ")
               displayBoard.display(board)
               game_on = False
           
           else:
               if checkSpace.fullSpace(board):
                   print("game has tied ")
                   displayBoard.display(board)
                   break
               else:
                    turn = 'player2'

        else:
            print("player second is going to play!")
            displayBoard.display(board)
            position = choosePosition.choose(board)
            placeMarker.placeMark(board, player2_marker, position)

            if winCheck.check(board, player2_marker):
               print("player_2 have won the match! ")
               displayBoard.display(board)
               game_on = False
           
            else:
                if checkSpace.fullSpace(board):
                   print("game has tied ")
                   displayBoard.display(board)
                   break
                else:
                    turn = 'player1'
    if not replay.replay():
        break