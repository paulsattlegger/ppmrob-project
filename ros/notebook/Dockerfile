ARG FROM_IMAGE=ros:humble
ARG OVERLAY_WS=/opt/ros/overlay_ws

# multi-stage for caching
FROM $FROM_IMAGE AS cacher

# copy manifests for caching
ARG OVERLAY_WS
WORKDIR $OVERLAY_WS/src/py_pubsub
ADD py_pubsub/package.xml .

# multi-stage for building
FROM $FROM_IMAGE AS builder

# install overlay dependencies
ARG OVERLAY_WS
WORKDIR $OVERLAY_WS
COPY --from=cacher $OVERLAY_WS/src ./src
RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    apt-get update && \
    apt-get install -y python3-pip && \
    rosdep install -y \
    --from-paths \
    src \
    --ignore-src \
    && rm -rf /var/lib/apt/lists/*

# copy overlay source
ADD py_pubsub ./src/py_pubsub

# build overlay source
ARG OVERLAY_MIXINS="release"
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
