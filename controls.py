import pydirectinput
import time

pydirectinput.PAUSE = 0
held_keys = set()

ACTIONS = {
        0: [],
        1: ["left"],
        2: ["right"],
        3: ["up"],
        4: ["down"],
        5: ["x"],
        6: ["up", "left"],
        7: ["up", "right"],
        8: ["down", "left"],
        9: ["down", "right"],
        }
MOVEMENTS = {"left", "right", "up", "down"}


def release_all():
    global held_keys
    for key in list(held_keys):
        pydirectinput.keyUp(key)
    held_keys.clear()


def perform_action(action):
    global held_keys
    keys = ACTIONS.get(action, [])
    desired = set(k for k in keys if k in MOVEMENTS)
    for key in held_keys - desired:
        pydirectinput.keyUp(key)
    for key in desired - held_keys:
        pydirectinput.keyDown(key)
    held_keys = desired
    if "x" in keys:
        pydirectinput.press("x")


def reset_controls():
    release_all()


def emergency_stop():
    release_all()