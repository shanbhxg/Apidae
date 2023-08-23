from beecolpy.beecolpy import abc
from sklearn.metrics.pairwise import cosine_similarity

# count = 0
# def func(x):
# 	global count
# 	count +=1
# 	print("x:",x)
# 	return x[0] + 100

def cosine_sim(user_tfidf, job_tfidf):
    # cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf, x),tfidf_jobid)
	csim = cosine_similarity(user_tfidf, job_tfidf)
	return csim

abc_obj = abc(cosine_sim, user_profile = , jd_list = , iterations=5, min_max='max', log_agents = True) #Load data
abc_obj.fit() #Execute the algorithm

print("total func calls:",cosine_sim)
#If you want to get the obtained solution after execute the fit() method:
solution = abc_obj.get_solution()
print("solution:",solution)

#If you want to get the number of iterations executed, number of times that
#scout event occur and number of times that NaN protection actuated:
iterations = abc_obj.get_status()[0]
print("number of iterations:",iterations)
# scout = abc_obj.get_status()[1]

# print("Scout:",scout)
# nan_events = abc_obj.get_status()[2]
# print("nan_events:",nan_events)

# #If you want to get a list with position of all points (food sources) used in each iteration:
# food_sources = abc_obj.get_agents()
# print("food :",food_sources)