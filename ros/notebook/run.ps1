$TAG = "ppmrob-ros-notebook:latest"

docker build --tag "$TAG" .
docker run --interactive --rm --tty `
    --volume="$(Get-Location):/ros2_ws/src" `
    --name ppmrob-ros-notebook `
    "$TAG"