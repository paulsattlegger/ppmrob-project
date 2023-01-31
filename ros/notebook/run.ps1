$TAG = 'ppmrob-ros-notebook:latest'

docker build --tag "$TAG" .
docker run --interactive --rm --tty `
    --name=ppmrob-ros-notebook `
    --volume="$(Get-Location):/ros2_ws" `
    "$TAG"