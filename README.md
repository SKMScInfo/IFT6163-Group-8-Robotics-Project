# IFT6163-Group-8-Robotics-Project
Compairson of IL, MBRL and SLAM with traditional planning


Les étapes pour faire le setup de ros2 et gazebo :- https://medium.com/@shubhjain_007/installing-ros2-humble-gazebo-on-mac-m1-m2-m3-silicon-853e0737dcc8

Version of gazebo used in the project: Gazebo Classic
It's an end of life version so we tried on Gazebo Ignition on a VM , but it crashes after installation due to hardware acceleration issues

Projet Outline:
1. Comparison of Dreamer and DADagger on the envrionments in Deepmind control suite vision benchmark for locomotion tasks , it suits our needs as it has expert trajectories readily availbale for multiple levels ( easy, medium and hard )

2. Comparation on navigation tasks
We are gonna test the simulations of gazebo with turtlebot. Basically for navigation tasks involving obstacle avoidance , it offers standard envrionments like turtlebot3_house, turtlebot3_world, willowgarage_world

3. Comparison of performance based on an equivalency:- IL and MBRL are offline and online learning tasks. So we need to see how to correlate the expert data Il is trained on and the number of trajectories used to train MBRL

4. Also we are using dreamerv3 isntead of dreamer as the latter as out of date libraries. So we need to update the paper with the new equations . ALso for DAdagger more mathematics needs to be explained 
