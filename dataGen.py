from sklearn.datasets import load_digits
import numpy as np
digits = load_digits()

print(digits.data.shape)

my_file = open('data.file',mode='a')
bdata = []
for i in range(digits.data.shape[0]):
	ldata = digits.data[i]
	cdata = [int(ele > 7) for ele in ldata]
	for ele in cdata:
		my_file.write(str(ele) + ' ')
	my_file.write('\n')
	bdata.append(cdata)

print(digits.data[0])
print(str(bdata[0]))
import matplotlib.pyplot as plt 
point = 9
plt.gray() 
plt.matshow(digits.images[point]) 
plt.matshow(np.array(bdata[point]).reshape(8,8))
plt.show() 
