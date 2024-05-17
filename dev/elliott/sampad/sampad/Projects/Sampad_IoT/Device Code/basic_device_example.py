from braingeneers.iot import messaging
import uuid


if __name__ == '__main__':
    mb = messaging.MessageBroker(str(uuid.uuid4))
    q=messaging.CallableQueue()
    mb.subscribe_message( topic="sampad_Device", callback=q )

    while True:
        topic_name , result_message = q.get()
        print( result_message )
