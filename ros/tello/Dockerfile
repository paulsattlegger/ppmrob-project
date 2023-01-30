FROM osrf/ros:foxy-desktop

RUN apt-get update && \
    apt-get install -y libasio-dev ros-foxy-cv-bridge ros-foxy-camera-calibration-parsers && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws/src

RUN git clone https://github.com/clydemcqueen/tello_ros.git && \
    git clone https://github.com/ptrmu/ros2_shared.git

WORKDIR /ros2_ws

RUN sh -c '. /opt/ros/foxy/setup.sh && colcon build --event-handlers console_direct+ --packages-skip tello_gazebo'

ADD ./docker-entrypoint.bash /

ENTRYPOINT ["/docker-entrypoint.bash"]

CMD ["/usr/bin/bash"]