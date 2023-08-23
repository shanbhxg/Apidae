from beecolpy.beecolpy import abc
from sklearn.metrics.pairwise import cosine_similarity

def cosine_sim(user_tfidf, job_tfidf):
    # cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf, x),tfidf_jobid)
	csim = cosine_similarity(user_tfidf, job_tfidf)
	return csim

jobs = [0.0830255831423132, 0.325325878549801, 0.10393778458475239, 0.06875841968245375, 0.05186689662716612, 0.10348202878666624, 0.13975459348701652, 0.07383646394440437, 0.08091323229027064, 0.0640663640330642, 0.17176245692528402, 0.07017312947757998, 0.03801211258714634, 0.051586316143992055, 0.06483098661281701, 0.325325878549801, 0.030689530104398247, 0.05565801153734859, 0.116133912727225, 0.10028648788924516, 0.06259738302859202, 0.223181432288947, 0.11274676277775546, 0.05534564428166872, 0.06140894348735485, 0.7154855301584309, 0.23849517671947698, 0.0287254275016071, 0.05084657228919795, 0.03262659681144902]
user = 0.6637384033843347

abc_obj = abc(cosine_sim, user_profile = user, jd_list = jobs, iterations=5, min_max='max', log_agents = True) #Load data
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