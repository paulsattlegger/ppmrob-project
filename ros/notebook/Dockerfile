ARG OVERLAY_WS=/opt/ros/overlay_ws
FROM ros:humble

# copy overlay source
ARG OVERLAY_WS
WORKDIR $OVERLAY_WS/src
ADD py_pubsub .

# install overlay dependencies
ARG OVERLAY_WS
WORKDIR $OVERLAY_WS
RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    apt-get update && \
    apt-get install -y python3-pip && \
    rosdep install -y \
    --from-paths \
    py_pubsub \
    --ignore-src \
    && rm -rf /var/lib/apt/lists/*

# build overlay source
ARG OVERLAY_MIXINS="release"
RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    colcon build \
    --packages-select \
    src/py_pubsub \
    --mixin $OVERLAY_MIXINS

# source entrypoint setup
ENV OVERLAY_WS $OVERLAY_WS
RUN sed --in-place --expression \
    '$isource "$OVERLAY_WS/install/setup.bash"' \
    /ros_entrypoint.sh

# run launch file
CMD ["bash"]
