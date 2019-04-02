# this module is used to check the winning of match

def check(board, mark):
    result = ((board[0] == mark and board[1] == mark and board[2] == mark) or
              (board[3] == mark and board[4] == mark and board[5] == mark) or
              (board[6] == mark and board[7] == mark and board[8] == mark) or
              (board[0] == mark and board[4] == mark and board[8] == mark) or
              (board[2] == mark and board[4] == mark and board[6] == mark) or
              (board[0] == mark and board[3] == mark and board[6] == mark) or
              (board[1] == mark and board[4] == mark and board[7] == mark) or
              (board[2] == mark and board[5] == mark and board[8] == mark))

    return result

              