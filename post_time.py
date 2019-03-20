import time
import sched
from datetime import datetime
import argparse
from post import *

FILE_PATH = ""

def run():
    res = post(load_file(FILE_PATH))
    print(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type=str)
    args = parser.parse_args()
    FILE_PATH = args.filepath

    s = sched.scheduler(time.time, time.sleep)
    start_time = datetime.strptime("2019/03/21 06:11", '%Y/%m/%d %H:%M')
    time_second = (start_time - datetime.now()).seconds
    s.enter(time_second, 1, run)
    s.run()

