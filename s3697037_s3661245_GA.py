from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys,time
from absl import app
from nasbench import api
import numpy as np
import nas_ioh
from numpy.random import default_rng
from numpy import random

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
seed = 23
ga_type = '+'
mu = 20
lmbda = 20
crossover_probabilty = 0.95
crossover_type = "uniform"
n_points = 2
mutation_rate = 1/21
def random_sampling_model(bench):
  while True:
    matrix = np.random.choice([0, 1], size=(7, 7))
    matrix = np.triu(matrix, 1)

    operations = [CONV1X1,CONV3X3,MAXPOOL3X3]
    ops = np.random.choice(operations,7)
    ops[0] = INPUT
    ops[6] = OUTPUT
    model_spec = api.ModelSpec(
      # Adjacency matrix of the module
      matrix=matrix,
      # Operations at the vertices of the module, matches order of matrix
      ops=list(ops))
    
    # check if the model is valid
    if bench.is_valid(model_spec):
      break
  
  x = np.empty(26,dtype=int)
  index = 0
  for i in range(7):
    for j in range(i+1,7):
      x[index] = matrix[i][j]
      index += 1
  for i in range (1,6):
    if ops[i] == CONV1X1:
      x[index] = 1
      index+=1
    elif ops[i] == CONV3X3:
      x[index] = 2
      index+=1
    elif ops[i] == MAXPOOL3X3:
      x[index] = 0
      index+=1
  return x


def uniform_crossover(parent1, parent2 ):

  offspring1 = []
  offspring2 = []

  for pos in range(len(parent1)):
    if(random.randint(2)>0):
      offspring1.append(parent1[pos])
      offspring2.append(parent2[pos])
    else:
      offspring1.append(parent2[pos])
      offspring2.append(parent1[pos])

  return offspring1, offspring2

def n_point_crossover(parent1,parent2, x = 1):
    offspring1 = np.copy(parent1)
    offspring2 = np.copy(parent2)
    rng = default_rng()
    crossover_points = rng.choice(len(parent1), size=x, replace=False)
    crossover_points = np.sort(crossover_points)
    low_limit = 0
    for index in range(len(crossover_points)):
        crossover_point = crossover_points[index]
        upper_limit = crossover_point
        if(index%2 == 0):
            offspring1[low_limit:upper_limit] = parent2[low_limit:upper_limit]
            offspring2[low_limit:upper_limit] = parent1[low_limit:upper_limit]
        
        if(index == len(crossover_points)-1):
            upper_limit = len(crossover_points)-1
            offspring1[low_limit:upper_limit] = parent2[low_limit:upper_limit]
            offspring2[low_limit:upper_limit] = parent1[low_limit:upper_limit]
    
        low_limit = crossover_point
        
    return  offspring1,offspring2

def mutation(element, mutation_rate = 0.2):
 
  for pos in range(21):
    if(np.random.rand() < mutation_rate):
      element[pos] = 1 - element[pos]
  for pos in range(21,26):
    if(np.random.rand() < mutation_rate):
      element[pos] = (element[pos] + 1 if np.random.rand()<0.5 else -1) % 3 
  return element

def main(argv):
  np.random.seed(seed)
  del argv  # Unused
  for r in range(nas_ioh.runs): # we execute the algorithm with 20 independent runs.
    budget = nas_ioh.budget
    
    f_opt = sys.float_info.min
    x_opt = None
    
    #generate initial random population
    parents = []
    parents_fit = []
    for i in range(mu):
        x = random_sampling_model(nas_ioh.nasbench)
        fitness = nas_ioh.f(x)
        parents.append(x)
        parents_fit.append(fitness)

    budget -= mu
    
    parents = np.array(parents)
    parents_fit = np.array(parents_fit)

    max_pos = np.argmax(parents_fit)
    x_opt = parents[max_pos]
    f_opt = parents_fit[max_pos]
    #print("best x:", x_opt,", f :",f_opt)
    #run the algorithm while we don't get a accuracy = 1, and we still have budget
    ok=0
    while (f_opt < 1 and budget > 0 and ok==0):
      offsprings = []
      offsprings_fit = []
      
      probabilities = (parents_fit - np.min(parents_fit))/ (np.sum(parents_fit) - np.min(parents_fit) * mu)
      
      
      while(budget > 0 and len(offsprings) < lmbda):
        if(np.isnan(probabilities).any() or np.sum(probabilities) != 1):
          parent_indexes = np.random.choice(int(mu),size=(2),replace=True)
        else:
          parent_indexes = np.random.choice(int(mu),size=(2),replace=True,p = probabilities)
        parent1 = parents[parent_indexes[0]]
        parent2 = parents[parent_indexes[1]]
        if(np.random.rand() < crossover_probabilty):
          if(crossover_type == "n_point"):
            offspring1 , offspring2 = n_point_crossover(parent1,parent2,n_points)
          else:
            offspring1 , offspring2 = uniform_crossover(parent1,parent2)
        else:
          offspring1, offspring2 = parent1, parent2    


        mutated_ofspring1 = mutation(offspring1,mutation_rate)
        mutated_ofspring2 = mutation(offspring2,mutation_rate)


        if(nas_ioh.is_valid(mutated_ofspring1)):
          fitness = nas_ioh.f(mutated_ofspring1)
          offsprings.append(mutated_ofspring1)
          offsprings_fit.append(fitness)
          budget -=1

        if(budget == 0 or not len(offsprings) < lmbda):
          break
        
        if(nas_ioh.is_valid(mutated_ofspring2)):
          fitness = nas_ioh.f(mutated_ofspring2)
          offsprings.append(mutated_ofspring2)
          offsprings_fit.append(fitness)
          budget -=1
      
      offsprings = np.array(offsprings)
      offsprings_fit = np.array(offsprings_fit)
      if(ga_type == "+"):        
        environment_pop = np.concatenate((parents,offsprings),axis = 0)
        environment_fit = np.concatenate((parents_fit,offsprings_fit))
      else:
        environment_pop = offsprings
        environment_fit = offsprings_fit

      max_pos = np.argmax(environment_fit)
      if(environment_fit[max_pos] >= f_opt):
        x_opt = environment_pop[max_pos]
        f_opt = environment_fit[max_pos]

      sorted_index = np.argsort(-environment_fit)
      environment_pop= environment_pop[sorted_index]
      environment_fit = environment_fit[sorted_index]

      parents = environment_pop[:mu]
      parents_fit = environment_fit[:mu]


      #print("best x:", x_opt,", f :",f_opt)
      



    print("run ", r, ", best x:", x_opt,", f :",f_opt)
    nas_ioh.f.reset() # Note that you must run the code after each independent run.



# If you are passing command line flags to modify the default config values, you
# must use app.run(main)
if __name__ == '__main__':
  start = time.time()
  app.run(main)
  end = time.time()
  print("The program takes %s seconds" % (end-start))