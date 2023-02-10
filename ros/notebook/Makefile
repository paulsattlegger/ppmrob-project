talker: build
	. /opt/ros/overlay_ws/install/setup.sh && \
	cd .. && \
	ros2 run tello_gate talker

listener: build
	. /opt/ros/overlay_ws/install/setup.sh && \
	cd .. && \
	ros2 run tello_gate listener

client: build
	. /opt/ros/overlay_ws/install/setup.sh && \
	cd .. && \
	ros2 run tello_gate client

build:
	. /opt/ros/humble/setup.sh && \
	cd .. && \
    colcon build \
    --packages-select \
    tello_gate tello_gate_msgs
