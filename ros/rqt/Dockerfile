FROM ros:humble

# install ros package
RUN apt-get update && \
    apt-get install -y ros-${ROS_DISTRO}-rqt* && \
    rm -rf /var/lib/apt/lists/*
