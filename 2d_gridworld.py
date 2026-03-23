# Source - https://stackoverflow.com/a/62617892
# Posted by WyattBlue, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-23, License - CC BY-SA 4.0

import keyboard  # using module keyboard
import time
import gymnasium
import gymnasium_env
import argparse

parser = argparse.ArgumentParser(description="2D Gridworld game!")

parser.add_argument("-s", "--size", type=int, help="Grid size", default=5)
parser.add_argument("-w", "--walls", type=bool, help="Static walls selector", default=False)
parser.add_argument("-dw", "--dwalls", type=bool, help="Dynamic walls selector", default=False)

args = parser.parse_args()

size = args.size
walls = args.walls
dwalls = args.dwalls

if walls or dwalls:
    print("Not implemented~")

env = gymnasium.make('gymnasium_env/GridWorld-v0', render_mode = "human", size = size)
env.reset()

m_map = {"W":3,"A":2,"S":1,"D":0}

print("WASD for movement (Press ESC to quit)")

score = 0

while True:
    # This blocks the code until a key is actually STRUCK (down and up)
    if (env.unwrapped._agent_location == env.unwrapped._target_location).all():
        break
    
    event = keyboard.read_event()
    
    if event.event_type == keyboard.KEY_DOWN:
        key = event.name.upper()
        if key in m_map:
            print(f"Action: {key}")
            env.step(m_map[key])
            score -= 0.01
        elif key == 'ESC':
            break

score += 1.01
print("Game success!")
print(f"score: {round(score,2)}")