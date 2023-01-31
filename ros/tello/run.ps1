$TAG = 'ppmrob-ros-tello:latest'

docker build --tag "$TAG" .
docker run --interactive --rm --tty `
    --name=ppmrob-ros-tello `
    "$TAG"