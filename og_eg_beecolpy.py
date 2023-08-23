from beecolpy.beecolpy import abc

def sphere(x):
	#print("x:",x)
	return x[0] + 10
	
abc_obj = abc(sphere, [(-10,10)]) #Load data
abc_obj.fit() #Execute the algorithm

#If you want to get the obtained solution after execute the fit() method:
solution = abc_obj.get_solution()
print("solution:",solution)

import random

integer_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

random_element = random.choice(integer_list)
print(random_element)