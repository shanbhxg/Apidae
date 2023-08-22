from abc.abc_problem import ABCProblem
from util.functions import FUNCTIONS

problem = ABCProblem(bees=10, function=FUNCTIONS['michalewicz'])
best_bee = problem.solve()
problem.replay()