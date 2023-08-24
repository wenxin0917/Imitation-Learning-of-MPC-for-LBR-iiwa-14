#!/usr/bin/env python3

import rospy
import numpy as np
import os
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

def trajectory_display(states):
    # Initialize ROS node
    rospy.init_node('trajectory_position_publisher', anonymous=True)

    # Define the joint names (update with your joint names)
    joint_names = ['iiwa14_joint_1', 'iiwa14_joint_2', 'iiwa14_joint_3', 'iiwa14_joint_4', 'iiwa14_joint_5', 'iiwa14_joint_6', 'iiwa14_joint_7']

    # Create publisher for initial joint positions
    joint_publisher = rospy.Publisher('/joint_states', JointState, queue_size=10)

    # Set the loop rate
    rate = rospy.Rate(5) 
    
    # create the JointState message
    joint_state = JointState()
    joint_state.name = joint_names
    
        
    # when the initial position is ready, send publish the torques
    while not rospy.is_shutdown():
        for i in range(np.shape(states)[0]):
            joint_state.header.stamp = rospy.Time.now()
            joint_state.position = states[i,0:7]
            joint_publisher.publish(joint_state)
            rate.sleep()
    

if __name__ == '__main__':
    try:
        
        # Get the path to the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Navigate one level up to reach the iiwa14 package directory
        pkg_dir = os.path.dirname(script_dir)
        
        # Construct the path to the .npy file inside the data directory
        # file_path = os.path.join(pkg_dir, 'data', 'state_single_point_with_wall_30.npy')
        # states = np.load(file_path)
        
        # load the circle joint reference .txt file
        file_path = os.path.join(pkg_dir, 'data', 'circle_state386_9.npy')
        states = np.load(file_path)
        
        
        # load the circle joint MPC result file
        # file_path = os.path.join(pkg_dir, 'data', 'N50.npy')
        # states = np.load(file_path)
        # Call the function to set initial position and publish the torques
        trajectory_display(states)
    except rospy.ROSInterruptException:
        pass
