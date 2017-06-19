import os
import sys
import glob
# scan the folder
root = sys.argv[1]
pattern = "*/*.jpg"

list_im = glob.glob(os.path.join(root, pattern))
list_im.sort()

with open('images.txt','w') as f:
    num = len(pattern.split('/'))
    for line in list_im:
        split = line.split('/')

        line = '/'.join(split[len(split)-num:])
        f.write(line + '\n')
