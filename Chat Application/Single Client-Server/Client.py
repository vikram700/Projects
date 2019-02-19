#import socket module
import socket

#take the hostname and port at which client want to send the request or communicate
host = '127.0.0.1' # address of server
port = 1234        # port of server

#Create the socket object for the client
client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)

# request is send for connection to the server
client_socket.connect((host,port))

# print function printed when we will connected with the server
print("Connect with server at  {}:{}".format(host,port))

# So now communication is begin now
while True :
    
    # client receive message and print message after decoding the message
    smsg = client_socket.recv(1024)
    print("\t\t\t\tserver-->",smsg.decode())
    
    #Now client type message for sending the message to the server and before send it will encode it.
    cmsg = input("cmsg-->")
    client_socket.send(cmsg.encode())
    
    # Whenever the client and server send or receive a bye message then communication will break
    if smsg.decode() == 'bye' or cmsg == 'bye' : 
        client_socket.close()
        break
