from braingeneers.iot import messaging
import uuid

###########################from my RPi3 code#################################
#start
import time
import os
import logging
import json
import datetime
import RPi.GPIO as GPIO
from time import sleep

# RPi 4 and 3B+ has same GPIO 
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
# valves 1-4
GPIO.setup(11, GPIO.OUT)
GPIO.output(11, GPIO.LOW)
GPIO.setup(13, GPIO.OUT)
GPIO.output(13, GPIO.LOW)
GPIO.setup(15, GPIO.OUT)
GPIO.output(15, GPIO.LOW)
GPIO.setup(16, GPIO.OUT)
GPIO.output(16, GPIO.LOW)
# valves 5-8
GPIO.setup(18, GPIO.OUT)
GPIO.output(18, GPIO.LOW)
GPIO.setup(12, GPIO.OUT)
GPIO.output(12, GPIO.LOW)
GPIO.setup(22, GPIO.OUT)
GPIO.output(22, GPIO.LOW)
GPIO.setup(29, GPIO.OUT)
GPIO.output(29, GPIO.LOW)
#valves 9-12
GPIO.setup(31, GPIO.OUT)
GPIO.output(31, GPIO.LOW)
GPIO.setup(33, GPIO.OUT)
GPIO.output(33, GPIO.LOW)
GPIO.setup(35, GPIO.OUT)
GPIO.output(35, GPIO.LOW)
GPIO.setup(37, GPIO.OUT)
GPIO.output(37, GPIO.LOW)
#valves 13-16
GPIO.setup(32, GPIO.OUT)
GPIO.output(32, GPIO.LOW)
GPIO.setup(36, GPIO.OUT)
GPIO.output(36, GPIO.LOW)
GPIO.setup(38, GPIO.OUT)
GPIO.output(38, GPIO.LOW)
GPIO.setup(40, GPIO.OUT)
GPIO.output(40, GPIO.LOW)

#mode defines here (maybe dictionary or enumate in the future)
MANUAL=1


# added two pins (2)
channel_num = 2
# the list indices (0~15) are according to the valve indices (1~16)
channel_pin = [11,13]

# When the mode is MANUAL, this function will be called. It changes status of
#	the pins according to the manual commands
def manual_control(param):
        i = 0;
        for channels in range(channel_num):
                if ((param>>((channel_num-1)-i))&1) >0:
                        GPIO.output(channel_pin[i], GPIO.HIGH)
                        print(i,"th LED is HIGH")
                else:
                        GPIO.output(channel_pin[i], GPIO.LOW)
                        print(i,"th LED is LOW")
                i+=1

# this is for reding message structure from the Json
def read_msg_struct(message):
	print("Message: ", message)
	mode = message['mode']
	param = message['param']
	if (mode == MANUAL):
		manual_control(param)
	else:
		print("Tick the manual box!!")
#end


if __name__ == '__main__':
    mb = messaging.MessageBroker(str(uuid.uuid4))
    q=messaging.CallableQueue()
    mb.subscribe_message( topic="Sampad_Device", callback=q )

    while True:
        received_message = q.get()
        print(received_message)
        read_msg_struct(received_message[1])
        print("in while loop")
