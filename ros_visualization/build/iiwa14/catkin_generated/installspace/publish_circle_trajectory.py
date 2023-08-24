#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
import os
import numpy as np

def publish_circle_trajectory(points):
    
    rospy.init_node('circle_trajectory_visualization', anonymous=True)
    marker_pub = rospy.Publisher('visualization', Marker, queue_size=10)
    rate = rospy.Rate(25)  # Set the publishing frequency to 50 Hz

    while not rospy.is_shutdown():
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "circle_trajectory"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        marker.scale.y = 0.01
        marker.scale.z = 0.01
        marker.color.a = 1.0 
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        

        for point in points:

            # Create a PointStamped message
            point_msg = Point()
            point_msg.x = point[0]
            point_msg.y = point[1]
            point_msg.z = point[2]
            marker.points.append(point_msg)

        # Publish the point
        marker_pub.publish(marker)
        rate.sleep()

if __name__ == '__main__':
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.dirname(script_dir)
    
    file_path = os.path.join(pkg_dir, 'data', '50circle_xy0.1_ee.txt')
    points = np.loadtxt(file_path,delimiter=',')
    
    try:
        publish_circle_trajectory(points)
    except rospy.ROSInterruptException:
        pass
