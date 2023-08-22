from beecolpy import abc

def sphere(x):
	y = 500
	print("x:",x)
	return x[0] + y

abc_obj = abc(sphere, [(-10,10)], iterations=5, min_max='min') #Load data
abc_obj.fit() #Execute the algorithm

#If you want to get the obtained solution after execute the fit() method:
solution = abc_obj.get_solution()
print(solution)
#If you want to get the number of iterations executed, number of times that
#scout event occur and number of times that NaN protection actuated:
# iterations = abc_obj.get_status()[0]
# scout = abc_obj.get_status()[1]
# nan_events = abc_obj.get_status()[2]

#If you want to get a list with position of all points (food sources) used in each iteration:
# food_sources = abc_obj.get_agents()