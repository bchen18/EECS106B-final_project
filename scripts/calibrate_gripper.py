#!/usr/bin/env python
import rospy
import sys

if sys.argv[1] == 'sawyer':
    from intera_interface import gripper as robot_gripper
else:
    from baxter_interface import gripper as robot_gripper

rospy.init_node('gripper_test')

# Set up the right gripper
right_gripper = robot_gripper.Gripper('right')

# # Calibrate the gripper (other commands won't work unless you do this first)
print('Calibrating...')
right_gripper.calibrate()
rospy.sleep(2.0)

# # Close the right gripper
print('Closing...')
right_gripper.close()
rospy.sleep(1.0)

# Open the right gripper
print('Opening...')
right_gripper.open()
rospy.sleep(2.0)
print('Done!')