import asyncio
import websockets
import json
import library

from library import Analyze_Sentiment
from library import recognize_intent
from library import extract_entities
from library import getMessageResponse 
 
from spacy.vocab import Vocab
from spacy.language import Language
nlp = Language(Vocab())

# Set of connected clients
connected_clients = set()



# Function to handle each client connection
async def handle_client(websocket):

    # Add the new client to the set of connected clients
    connected_clients.add(websocket)
    print(f"Yeni client bağlandı")
    try:
        # Listen for messages from the client
        async for message in websocket:
            print(f"Yeni mesaj alındı")
            # Broadcast the message to all other connected clients
            for client in connected_clients:
                if client == websocket:
                    #message= message + " sunucu yanıtıdır"
                    sentiment= Analyze_Sentiment(message)
                    intent= recognize_intent(message)
                    entitites= extract_entities(message)
                    responseLemma=getMessageResponse(message)
                    responseObject={
                        "Sentiment":sentiment,
                        "Intent":intent,
                        "Entities":entitites,
                        "Lemma":responseLemma
                    }
                    await client.send(json.dumps(responseObject))
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        # Remove the client from the set of connected clients
        connected_clients.remove(websocket)

# Main function to start the WebSocket server
async def main():
    server = await websockets.serve(handle_client, 'localhost', 12345)
    await server.wait_closed()

# Run the server
if __name__ == "__main__":
    asyncio.run(main())