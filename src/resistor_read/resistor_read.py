#!/usr/bin/env python
import rospy
import serial

#Import the String message type from the /msg directory of
#the std_msgs package.
from std_msgs.msg import Float32

if __name__ == '__main__':
    ser = serial.Serial('/dev/ttyACM0')
    rospy.init_node('force_sensor', anonymous=True)
    pub = rospy.Publisher('force_resistance', Float32, queue_size=10)
    r = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        ser_bytes = ser.readline()
        pub.publish(float(ser_bytes))