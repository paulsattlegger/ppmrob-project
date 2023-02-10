ARG FROM_IMAGE=osrf/ros:humble-desktop
ARG OVERLAY_WS=/opt/ros/overlay_ws

FROM $FROM_IMAGE

# copy manifest for caching
ARG OVERLAY_WS
WORKDIR $OVERLAY_WS/src/tello_gate
ADD tello_gate/package.xml .
ARG OVERLAY_WS
WORKDIR $OVERLAY_WS/src/tello_gate_msgs
ADD tello_gate_msgs/package.xml .

# install overlay dependencies
ARG OVERLAY_WS
WORKDIR $OVERLAY_WS
# https://github.com/moby/buildkit/blob/master/frontend/dockerfile/docs/reference.md#run---mount
RUN \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=cache,target=/root/.cache/pip \
    . /opt/ros/$ROS_DISTRO/setup.sh && \
    apt-get update && \
    apt-get install -y \
    python3-pip && \
    rosdep install -y \
    --from-paths \
    src \
    --ignore-src

# copy overlay source
ARG OVERLAY_WS
WORKDIR $OVERLAY_WS/src/tello_gate
ADD tello_gate .
ARG OVERLAY_WS
WORKDIR $OVERLAY_WS/src/tello_gate_msgs
ADD tello_gate_msgs .

# build overlay source
ARG OVERLAY_MIXINS="release"
WORKDIR $OVERLAY_WS
RUN . /opt/ros/$ROS_DISTRO/setup.sh && \
    colcon build \
    --packages-select \
    tello_gate tello_gate_msgs \
    --mixin $OVERLAY_MIXINS

# source entrypoint setup
ENV OVERLAY_WS $OVERLAY_WS
RUN sed --in-place --expression \
    '$isource "$OVERLAY_WS/install/setup.bash"' \
    /ros_entrypoint.sh
