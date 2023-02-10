replay: build
	. /opt/ros/overlay_ws/install/setup.sh && \
	cd .. && \
	ros2 run tello_gate replay

buffer: build
	. /opt/ros/overlay_ws/install/setup.sh && \
	cd .. && \
	ros2 run tello_gate buffer

inference: build
	. /opt/ros/overlay_ws/install/setup.sh && \
	cd .. && \
	ros2 run tello_gate inference

build:
	. /opt/ros/humble/setup.sh && \
	cd .. && \
    colcon build \
    --packages-select \
    tello_gate tello_gate_msgs
