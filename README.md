# Apidae : A Personalized Job Recommendation System
![image](https://github.com/shanbhxg/Apidae/assets/76104354/97a26f1d-1dda-462f-8ea8-aa8772f9f041)

class abc: \
functions:<br> 
init<br>
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
generate_food_source(self, index) -> self.abc.foods[index] = _FoodSource(self.abc, self) then execute_nan_protection(index)<br><br>

prob_i(self, actual_fit, max_fit) -> return 0.9*(actual_fit/max_fit) + 0.1 \
calculate_fit(self, evaluated_position) -> converts cost_func to fit_func by doing min or max<br>
food_source_dance(self, index) -> Generate a partner food source to generate a neighbor point to evaluate<br><br>

employer_bee_phase(self) -> for loop calls food_source_dance<br>
onlooker_bee_phase(self)<br>
scout_bee_phase(self) -> Generate up to one new food source that does not improve over scout_limit evaluation tries<br>
memorize_best_solution(self) -> self.abc.best_food_source = self.abc.foods[best_food_index]<br>






          
