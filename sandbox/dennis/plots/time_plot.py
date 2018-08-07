import numpy as np
import matplotlib.pyplot as plt

N = 3
sampletime = [42.63] * 3
processtime = [0.58] * 3
updatetime = [1.23, 13.96, 7.77]

width = 0.35
ind = np.arange(N)

sample = plt.bar(ind, sampletime, width)
process = plt.bar(ind, processtime, width, bottom=sampletime, color='orange')
update = plt.bar(ind, updatetime, width, bottom=[sum(q) for q in zip(processtime, sampletime)], color='green')

plt.ylabel('Time (seconds per iteration)')
plt.title('Algorithm runtimes')
plt.xticks(ind + width/2, ('VPG', 'TRPO', 'PPO'))
plt.yticks(np.arange(0, 71, 10))
plt.legend((sample[0], process[0], update[0]), ('Sample', 'Process', 'Update'))

plt.savefig('Timing.png')
