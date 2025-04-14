import socket

HOST = "127.0.0.1"  # Sunucu adresi
PORT = 12345        # Aynı portu kullanmalı 

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

print("Sunucuya bağlanıldı! Mesaj gönderin ('exit' yazarak çıkabilirsiniz).")

while True:
    message = input("Mesajınız: ")
    if message.lower() == "exit":  # 'exit' yazılırsa çık
        break
    client_socket.sendall(message.encode())  # Mesajı gönder
    data = client_socket.recv(1024)  # Sunucudan cevap al
    print(f"Sunucudan gelen cevap: {data.decode()}")

client_socket.close()
print("Bağlantı kapatıldı.")