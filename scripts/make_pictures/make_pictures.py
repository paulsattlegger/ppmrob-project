import itertools
from pathlib import Path
from time import sleep

import cv2
from djitellopy import Tello


def current_count():
    return max([int(p.stem) for p in Path(__file__).parent.joinpath("pictures").iterdir()])


def main():
    tello = Tello()
    tello.connect()
    tello.streamon()

    for i in itertools.count(current_count() + 1):
        frame_read = tello.get_frame_read()
        cv2.imwrite(f"{i}.png", frame_read.frame)  # noqa
        sleep(1 / 30)


if __name__ == "__main__":
    main()
