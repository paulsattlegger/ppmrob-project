SHELL=/usr/bin/bash

ppmrob-project-11810278-11810787.zip: data.tar.xz
	git archive -o $@ --prefix='ppmrob-project-11810278-11810787/' --add-file=$< HEAD

data.tar.xz:
	tar --exclude='*.zip' --exclude='*.gitkeep' -cvJf $@ torch/data/ ros/data/model/ ros/data/images/picture5_{0..50}.png
