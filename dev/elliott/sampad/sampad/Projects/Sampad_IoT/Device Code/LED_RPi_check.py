from braingeneers.iot import messaging
import uuid

def read_message(topic_name, result_message):
	print( result_message )

if __name__ == '__main__':
    mb = messaging.MessageBroker(str(uuid.uuid4))
    q=messaging.CallableQueue()
    mb.subscribe_message( topic="Sampad_Device", callback=q )

    while True:
        topic_name , result_message = q.get()
        read_message(topic_name, result_message)
        print("in while loop")
