# Source - https://stackoverflow.com/a/62617892
# Posted by WyattBlue, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-23, License - CC BY-SA 4.0

import sys
import tty
import termios
import gymnasium
import gymnasium_env
import argparse

parser = argparse.ArgumentParser(description="2D Gridworld game!")

parser.add_argument("-l", "--level", type=int, help="Curriculum level (0-4)", default=0)

args = parser.parse_args()

env = gymnasium.make('gymnasium_env/GridWorld-v0', render_mode="human", level=args.level)
env.reset()

m_map = {"W":3,"A":2,"S":1,"D":0}

print("WASD for movement (Press ESC to quit)")

score = 0

def read_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch

while True:
    if (env.unwrapped._agent_location == env.unwrapped._target_location).all():
        break

    ch = read_key()
    key = ch.upper()

    if ch == '\x1b':  # ESC
        break
    elif key in m_map:
        print(f"Action: {key}")
        env.step(m_map[key])
        score -= 0.01

score += 1.01
print("Game success!")
print(f"score: {round(score,2)}")