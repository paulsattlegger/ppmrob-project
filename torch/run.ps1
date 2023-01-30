docker run --gpus all `
    --interactive --tty `
    --volume="$(Get-Location):/workspace/ppmrob" `
    --name=ppmrob-torch `
    --ipc=host `
    nvcr.io/nvidia/pytorch:22.12-py3