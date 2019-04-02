# This function is used for deciding to play game again or not
def replay():
    val = input("enter you want to play again or not --> yes or no :").lower()
    if val not in ['yes', 'no']:
        replay()
    else:
        if(val == 'yes'):
            return True
        else:
            return False
    