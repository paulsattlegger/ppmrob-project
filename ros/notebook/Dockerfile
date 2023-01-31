ARG FROM_IMAGE=ros:humble
ARG OVERLAY_WS=/opt/ros/overlay_ws

FROM $FROM_IMAGE

# copy manifests for caching
ARG OVERLAY_WS
WORKDIR $OVERLAY_WS/src/py_pubsub
ADD py_pubsub/package.xml .

# install overlay dependencies
ARG OVERLAY_WS
WORKDIR $OVERLAY_WS
RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    apt-get update && \
    apt-get install -y python3-pip && \
    rosdep install -y \
    --from-paths \
    src \
    --ignore-src \
    && rm -rf /var/lib/apt/lists/*

# copy overlay source
ARG OVERLAY_WS
WORKDIR $OVERLAY_WS/src/py_pubsub
ADD py_pubsub .

# build overlay source
ARG OVERLAY_MIXINS="release"
ARG OVERLAY_WS
WORKDIR $OVERLAY_WS
RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    colcon build \
    --packages-select \
    py_pubsub \
    --mixin $OVERLAY_MIXINS

# source entrypoint setup
ENV OVERLAY_WS $OVERLAY_WS
RUN sed --in-place --expression \
    '$isource "$OVERLAY_WS/install/setup.bash"' \
    /ros_entrypoint.sh
