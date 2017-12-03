import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

inpt = pd.read_csv("../input/training_variants")
inpt = inpt['Class']
inpt = list(inpt)
u,c = np.unique(inpt,return_counts=True)
input_dict = dict(zip(u,c))

out = pd.read_csv('./submission_all.csv')
out = out.drop('id',axis=1)
out = out.T

lookup = {'class1' : 1, 'class2' : 2, 'class3' : 3, 'class4' : 4,
		  'class5' : 5, 'class6' : 6, 'class7' : 7, 'class8' : 8,
		  'class9' : 9}

output = []
for i in range(len(out.T)):
	output.append(lookup[np.argmax(out[i])])

u,c = np.unique(output,return_counts=True)
output_dict = dict(zip(u,c))

x = 0
for e in output_dict.values():
	x += e

y =	0
for e in input_dict.values():
	y += e
	
factor = y/x

for key in output_dict.keys():
	output_dict[key] = output_dict[key] * factor

out_vals = []
for i in range(1,10):
	if i not in output_dict:
		output_dict[i] = 0
	out_vals.append(output_dict[i])


print(output_dict)

x = range(1,10)
y = list(x)
in_bars = []
out_bars = []
for i in x:
	in_bars.append(i-0.1)
	out_bars.append(i+0.1)
	


ax = plt.subplot(111)
ax.bar(in_bars,input_dict.values(),width=0.2,color='b',align='center')
ax.bar(out_bars,out_vals,width=0.2,color='r',align='center')
plt.show()

