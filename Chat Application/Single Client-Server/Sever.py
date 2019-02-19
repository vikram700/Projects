# coding for server

#import the socket module
import socket

#create an object of the socket for server
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Take imformation about the host and the port at which server is listening
host = '127.0.0.1'
port = 1234

# Bind the network interface and port no to the server_socket object
#Server can listen an request at particular port
server_socket.bind((host, port))
print("server socket is binded at address {} and port {} ".format(host,port))

# ready to server object for listening
server_socket.listen()

# code for accepting the client
client_socket, client_addr = server_socket.accept()
print("client is connected to the server with address and port as {} and {} ".format(*client_addr))

while True:
    
    # server will type messase for client so server take input
    smsg = input("Server --->")
    client_socket.send(smsg.encode()) #we need to decode the message because message is send in bytecode
    
    
    # Here server receive message from client
    cmsg = client_socket.recv(1024)
    print("Client ---> ",cmsg.decode()) #before printing we need to decode it because data in byte format
    
    
    # Commuication between the client and server will be stop if any one of them send message "bye"
    if cmsg.decode() == 'bye' or smsg == 'bye':
        client_socket.close()
        server_socket.close()
        break
