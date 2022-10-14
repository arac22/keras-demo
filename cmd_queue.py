#-------------------------------------------------------------------------------
# Name:        modulo1
# Purpose:
#
# Author:      Alberto
#
# Created:     14/10/2022
# Copyright:   (c) Alberto 2022
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from collections import deque

def main():

    cmd = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,]
    vel = [0,0,0,0,0,]

    cmd_queue = deque(cmd)
    vel_queue = deque(vel)

    for i in range(1,20):

        vel_queue.appendleft(i)
        vel_queue.pop()

        cmd_queue.appendleft(i)
        cmd_queue.pop()

        print('STEP:', i)
        print('VEL:', vel_queue[0], vel_queue[1],vel_queue[2],vel_queue[3],vel_queue[4])
        print('CMD:', cmd_queue[0], cmd_queue[3],cmd_queue[6],cmd_queue[9],cmd_queue[12])
        print('\n')


    pass

if __name__ == '__main__':
    main()
