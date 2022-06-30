import os
from subprocess import Popen
from time import sleep
import psutil

usage = {"CPU": [], "Mem": []}

total_bl, used_bl, free_bl = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
cpu_bl = psutil.cpu_percent(interval=3)

p = Popen(["bash", "/data/anand/Automorph/run.sh"])

while p.poll() is None: 
  cpu_usage = psutil.cpu_percent(interval=1)
  total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])

  usage['CPU'].append(float(cpu_usage-cpu_bl))
  usage['Mem'].append(float(used_memory-used_bl)) 
  sleep(5)


with open('resources.csv', 'w') as f:
  f.write("cpu,mem\n")
  for i in range(len(usage["CPU"])):
    f.write("{0},{1}\n".format((usage["CPU"][i]), usage["Mem"][i]))
