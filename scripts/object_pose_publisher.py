#!/usr/bin/env python  
"""
Object Publisher script for Project 3.
Produces object poes relative to AR tag.
Usage:
  rosrun proj3_pkg object_pose_publisher.py baxter
or
  rosrun proj3_pkg object_pose_publisher.py sawyer
This will be taken care of you in a launch file.
Author: Chris Correa
"""
import rospy
import sys
import numpy as np
import tf
import utils

class ObjectTemplate(object):
    def __init__(self, name, ar_marker, R_ar_obj=np.eye(3), t_ar_obj=np.zeros(3)):
        """
        Struct for specifying object templates

        Parameters
        ----------
        name : string
            name of object
        ar_marker : string
            name of ar marker on object template
        R_ar_obj : 3x3 :obj:`numpy.ndarray`
            rotation between AR marker and object
        t_ar_obj : 3x' :obj:`numpy.ndarray`
            translation between AR marker and objectSS
        """
        self.name = name
        self.ar_marker = ar_marker
        self.T_ar_obj = utils.create_transform_matrix(R_ar_obj, t_ar_obj)
        self.translation_vector = t_ar_obj

    @property
    def q_ar_obj(self):
        """
        Returns the rotation between the AR marker and the object in quaternion form
        """
        return tf.transformations.quaternion_from_matrix(self.T_ar_obj)

    @property
    def t_ar_obj(self):
        """
        Returns the translation between the AR marker and the object
        """
        if len(self.translation_vector) != 3:
            raise ValueError("Translation must be specified as a 3-vector")
        return self.translation_vector

OBJECT_TEMPLATES = {
    ObjectTemplate(name='pawn', ar_marker='ar_marker_9', t_ar_obj=[-0.080, 0.140, 0.045]),
    ObjectTemplate(name='nozzle', ar_marker='ar_marker_10', t_ar_obj=[-0.100, 0.160, 0.0206])
}

if __name__ == '__main__':
    """
    This node publishes the transform between the ar marker and the object.  The purpose of this
    is so that you can call lookup_transform('bar_clamp', 'world'), and not have to do transformation
    matrix multiplications.  
    """

    rospy.init_node('object_pose_publisher')
    robot = sys.argv[1]
    camera_frame = ''
    if robot == 'baxter':
        camera_frame = 'left_hand_camera'
    elif robot == 'sawyer':
        camera_frame = 'usb_cam'
    else:
        print("Unknown robot type!")
        rospy.shutdown()

    broadcaster = tf.TransformBroadcaster()
    listener = tf.TransformListener()
 
    print('Publishing object pose')
    
    rate = rospy.Rate(1.0)
    while not rospy.is_shutdown():
        for object_template in OBJECT_TEMPLATES:
	        try:
	            broadcaster.sendTransform(
	                object_template.t_ar_obj, 
	                object_template.q_ar_obj, 
	                listener.getLatestCommonTime('base', camera_frame), 
	                object_template.name, 
	                object_template.ar_marker
	            )
	        except:
	            continue
        rate.sleep()