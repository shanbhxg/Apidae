# Apidae : A Personalized Job Recommendation System
![image](https://github.com/shanbhxg/Apidae/assets/76104354/97a26f1d-1dda-462f-8ea8-aa8772f9f041)

class abc: \
functions: \ 
init \
          fit(self) -> Returns a list with values found as minimum/maximum coordinate. \
          get_agents(self, reset_agents=False) -> Returns a list with the position of each food source during each iteration. \
          get_solution(self) -> Returns the value obtained after fit() the method. \
          get_status(self) -> tuple with iters,scouts,NaN protection info  

class _FoodSource \
functions: \
init \
evaluate_neighbor(self, partner_position) -> random number generation, update self_point to neighbour_point if its more fit \

class _ABC_engine \
functions: \
init -> sets self,abc = abc \
check_nan_lock(self) -> error for infinite loop \
execute_nan_protection(self, food_index) -> self.abc.foods[food_index] = _FoodSource(self.abc, self) \
generate_food_source(self, index) -> self.abc.foods[index] = _FoodSource(self.abc, self) then execute_nan_protection(index)  



          
