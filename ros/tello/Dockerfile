ARG FROM_IMAGE=osrf/ros:foxy-desktop
ARG OVERLAY_WS=/opt/ros/overlay_ws

# multi-stage for caching
FROM $FROM_IMAGE AS cacher

# clone overlay source
ARG OVERLAY_WS
WORKDIR $OVERLAY_WS/src
COPY overlay.repos ../overlay.repos
RUN vcs import ./ < ../overlay.repos

# copy manifests for caching
WORKDIR /opt
RUN mkdir -p /tmp/opt && \
    find ./ -name "package.xml" | \
    xargs cp --parents -t /tmp/opt && \
    find ./ -name "COLCON_IGNORE" | \
    xargs cp --parents -t /tmp/opt || true

# multi-stage for building
FROM $FROM_IMAGE AS builder

# install overlay dependencies
ARG OVERLAY_WS
WORKDIR $OVERLAY_WS
COPY --from=cacher /tmp/$OVERLAY_WS/src ./src

RUN apt-get update && \
    apt-get install -y libasio-dev && \
    rosdep install -y \
    --from-paths \
    src/clydemcqueen/tello_ros/tello_driver \
    src/clydemcqueen/tello_ros/tello_msgs \
    src/clydemcqueen/tello_ros/tello_description \
    src/ptrmu/ros2_shared \
    --ignore-src \
    && rm -rf /var/lib/apt/lists/*

# build overlay source
COPY --from=cacher $OVERLAY_WS/src ./src
ARG OVERLAY_MIXINS="release"
RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    colcon build \
    --packages-select \
    tello_driver \
    tello_msgs \
    tello_description \
    ros2_shared \
    --mixin $OVERLAY_MIXINS \
    --event-handlers console_direct+

# source entrypoint setup
ENV OVERLAY_WS $OVERLAY_WS
RUN sed --in-place --expression \
    '$isource "$OVERLAY_WS/install/setup.bash"' \
    /ros_entrypoint.sh

# run launch file
CMD ["ros2", "launch", "tello_driver", "teleop_launch.py"]
