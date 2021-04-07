import time
from tqdm import *
import sys

# for i in tqdm(range(5), position=0, leave=True):
#     for j in tqdm(range(5), position = 1, leave=True):
#         time.sleep(.1)

with trange(10, position=0, desc='Outter', bar_format='{desc:15}{percentage:3.0f}%|{bar:50}{r_bar}') as outter_range:
    for i in outter_range:

        if i == 5:
            break
        outter_range.set_description(str(i))
        leave = i == len(outter_range) - 1
        for _ in trange(10, position=1, leave=leave, desc='Inner', bar_format='{desc:15}{percentage:3.0f}%|{bar:50}{r_bar}'):
            time.sleep(.1)
            # print("test", file=sys.stderr)

print("done")