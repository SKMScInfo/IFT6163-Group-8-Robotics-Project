# IFT6163-Group-8-Robotics-Project
Compairson of IL, MBRL and SLAM with traditional planning

The steps to setup ROS2 and Gazebo :- https://medium.com/@shubhjain_007/installing-ros2-humble-gazebo-on-mac-m1-m2-m3-silicon-853e0737dcc8

Version of gazebo used in the project: Gazebo Classic
It's an end of life version so we tried on Gazebo Ignition on a VM, but it crashes after installation due to hardware acceleration issues

Project Outline:
1. Comparison of Dreamer and DADAgger on the envrionments in Deepmind control suite vision benchmark for locomotion tasks, it suits our needs as it has expert trajectories readily availbale for multiple levels (easy, medium and hard)

2. Comparison on navigation tasks
We are gonna test the simulations of gazebo with turtlebot. Basically for navigation tasks involving obstacle avoidance, it offers standard envrionments like turtlebot3_house, turtlebot3_world, willowgarage_world

3. Comparison of performance based on an equivalency:
    - IL and MBRL are offline and online learning tasks. So we need to see how to correlate the expert data IL is trained on and the number of trajectories used to train MBRL

4. Also we are using dreamerv3 isntead of dreamer as the latter as out of date libraries. So we need to update the paper with the new equations. Also for DADAgger more mathematics needs to be explained.


The following setup is to be followed for making the project work on MacOS
1. Install UTM
2. Add Ubuntu 22.02 image to UTM ( use virtio-gpu-gl-pci in the display settings when running the VM, it supports GPU, also may have to wait for 10 secons for the display output to come )
3. Install ros2-humble-desktop and ignition fortess binary installations  following  the official pages
4. Install turblebot4 from here https://turtlebot.github.io/turtlebot4-user-manual/software/turtlebot4_simulator.html
5. To run SLAM for out setup do :- LIBGL_ALWAYS_SOFTWARE=1 ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py slam:=true nav2:=true rviz:=true 
It's important to use LIBGL_ALWAYS_SOFTWARE=1, as otherwise OPENGL was cause errors because of 2.3 version being needed, but VMs on MACOS only supporting till 2.1





