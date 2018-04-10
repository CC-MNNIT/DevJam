import json
from itertools import chain

# Script to create Q-Value JSON file, initilazing with zeros

qval = {}
# X -> [-40,-30...120] U [140, 210 ... 490]
# Y -> [-300, -290 ... 160] U [180, 240 ... 420]
qval["-20_-240_4"] = [0,0]


fd = open('qval.json', 'w')
json.dump(qval, fd)
fd.close()
