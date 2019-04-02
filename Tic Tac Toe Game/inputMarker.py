def takeInput():
    marker = ""

    while marker not in ('X','O'):
        marker = input("Enter your marker X or O? : ")

    if(marker == 'X'):
        return ('X','O')
    else:
        return ('O','X')
