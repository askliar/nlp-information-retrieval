import numpy as np
def gain(a):
	idx = np.argwhere(a == 0)[0][0]
	if idx == 0:
		g = 0
	else:
		g = 2**idx
	return g
l = []
for i in range(10000):
	s = np.random.binomial(size=1000, n=1, p= 0.5)
	l.append(gain(s))

print(l)
print(sum(l) / len(l))
