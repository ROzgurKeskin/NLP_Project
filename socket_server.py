import socket
import threading

HOST = "127.0.0.1"  # Yerel ağda çalışması için localhost
PORT = 12345        # Kullanılacak port 

# Socket oluşturma
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(10)  # Maksimum 1 bağlantı kabul edecek

print(f"Sunucu {HOST}:{PORT} adresinde çalışıyor...")

def handleClient(conn,addr):
# Bağlantıyı kabul et
    print(f"Bağlantı alındı: {addr}")

    while True:
        try:
            data = conn.recv(1024)  # 1024 byte veri al
            if not data:
                print("İstemci bağlantıyı kapattı.")
                break  # Bağlantı kesildiğinde döngüden çık
            print(f"İstemciden gelen: {data.decode()}")
            serverResponse=f"{data.decode()} sunucu yanıtıdır"
            conn.sendall(serverResponse.encode())  # Geri bildirim gönder
        except ConnectionResetError:
            print("İstemci aniden bağlantıyı kesti.")
            break

    conn.close()

while True:
    conn, addr = server_socket.accept()  # Yeni bağlantıyı kabul et
    client_thread = threading.Thread(target=handleClient, args=(conn, addr))
    client_thread.start()  # Her istemciyi ayrı bir thread'de çalıştır

server_socket.close()