
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys,time
from absl import app
from nasbench import api
import numpy as np
import nas_ioh

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'

step_type = "n-step"
mu = 10
ro = 10
lmbda = 10
es_type = "+"
connections = 21
nodes = 5
types_of_nodes = 3
thao_0 = 1/ np.sqrt(35)
thao_g = 1/ np.sqrt(35)
thao = 1/np.sqrt(2* np.sqrt(35))
seed = 23
min_value = 0
max_value = 1



def softmax_max_pos(x):
  prob = np.exp(x)/np.sum(np.exp(x))
  return np.argmax(prob)

def decode_solution(x):
  decoded_x = []
  for pos in range(connections):
    decoded_x.append(np.int16(x[pos]+0.5))

  for pos in range(connections, connections + types_of_nodes * nodes, types_of_nodes):
    decoded_x.append(softmax_max_pos(np.array(x[pos : (pos + types_of_nodes)])))

  return np.array(decoded_x)

def random_sampling_model():
  
  while True:
    x = np.random.uniform(min_value, max_value,  connections + types_of_nodes * nodes)
    decoded_x = decode_solution(x)
    if(nas_ioh.is_valid(decoded_x)):
      break

  if(step_type == "one-step"):
    sigma = np.random.uniform(0, max_value - min_value)
  elif(step_type == "n-step"):
    sigma = np.random.uniform(0, max_value - min_value, connections + types_of_nodes * nodes)
  
  # print(np.shape(sigma))
  return x,sigma


def recombination(parents,sigmas):
  offspring = np.mean(parents, axis = 0)
  offspring_sigma = np.mean(sigmas, axis = 0)
  return offspring, offspring_sigma


def mutation(element, sigma):
  if(step_type == "one-step"):
    sigma = sigma * np.exp(np.random.normal(0,thao_0))
    for pos in range(len(element)):
      new_value = element[pos] + np.random.normal(0,sigma)
      if(pos < connections):
        if(new_value > 1):
          new_value = 1
        if(new_value < 0):
          new_value = 0
        
      element[pos] = new_value
      
  elif(step_type == "n-step"):
    global_normal = np.random.normal(0,thao_g)
    for pos in range(len(sigma)):
      sigma[pos] = sigma[pos] * np.exp(global_normal + np.random.normal(0,thao))
      new_value = element[pos] + np.random.normal(0,sigma[pos])
      if(pos < connections):
        if(new_value > 1):
          new_value = 1
        if(new_value < 0):
          new_value = 0
      element[pos] = new_value
  
  return element,sigma

def main(argv):
  del argv  # Unused
  np.random.seed(seed)
  for r in range(nas_ioh.runs): # we execute the algorithm with 20 independent runs.
    budget = nas_ioh.budget
    
    f_opt = sys.float_info.min
    x_opt = None
    
    #generate initial random population
    parents = []
    parents_sigma = []
    parents_fit = []
    for _ in range(mu):
        x,sigma = random_sampling_model()
        decoded_x = decode_solution(x)
        # print(decoded_x)
        fitness = nas_ioh.f(decoded_x)
        parents.append(x)
        parents_sigma.append(sigma)
        parents_fit.append(fitness)

    budget -= mu
    
    parents = np.array(parents)
    parents_sigma = np.array(parents_sigma)
    parents_fit = np.array(parents_fit)

    max_pos = np.argmax(parents_fit)
    x_opt = parents[max_pos]
    sigma_opt = parents_sigma[max_pos]
    f_opt = parents_fit[max_pos]
    # print("best x:", x_opt, "best_x_decoded", decode_solution(x_opt), "sigma:", sigma_opt, "f :", f_opt)
    #run the algorithm while we don't get a accuracy = 1, and we still have budget
    
    while (f_opt < 1 and budget > 0):
      offsprings = []
      offsprings_sigma = []
      offsprings_fit = []
      
      
      while(budget > 0 and len(offsprings) < lmbda):
        parent_indexes = np.random.choice(int(mu),size=(ro),replace=True)
        
        parents_for_cross = parents[parent_indexes]
        sigmas_for_mutation = parents_sigma[parent_indexes]


        offspring, offspring_sigma = recombination(parents_for_cross, sigmas_for_mutation)
        

        mutated_offspring, mutated_sigma = mutation(offspring,offspring_sigma)
        decode_offspring = decode_solution(mutated_offspring)


        if(nas_ioh.is_valid(decode_offspring)):
          fitness = nas_ioh.f(decode_offspring)
          offsprings.append(mutated_offspring)
          offsprings_sigma.append(mutated_sigma)
          offsprings_fit.append(fitness)
          budget -= 1
      
      offsprings = np.array(offsprings)
      offsprings_sigma = np.array(offsprings_sigma)
      offsprings_fit = np.array(offsprings_fit)

      if(es_type == "+"):
        environment_pop = np.concatenate((parents,offsprings),axis = 0)
        environment_sigma = np.concatenate((parents_sigma,offsprings_sigma),axis = 0)
        environment_fit = np.concatenate((parents_fit,offsprings_fit))
      elif(es_type == ","):
        environment_pop = offsprings
        environment_sigma = offsprings_sigma
        environment_fit = offsprings_fit
      
      max_pos = np.argmax(environment_fit)
      if(environment_fit[max_pos] >= f_opt):
        x_opt = environment_pop[max_pos]
        sigma_opt = environment_sigma[max_pos]
        f_opt = environment_fit[max_pos]

      sorted_index = np.argsort(-environment_fit)
      environment_pop= environment_pop[sorted_index]
      environment_sigma = environment_sigma[sorted_index]
      environment_fit = environment_fit[sorted_index]

      parents = environment_pop[:mu]
      parents_sigma = environment_sigma[:mu]
      parents_fit = environment_fit[:mu]


      # print("best x:", x_opt, "best_x_decoded", decode_solution(x_opt), "sigma:", sigma_opt, "f :", f_opt)
      # break



    print("run ", r)
    # print("best x:", x_opt)
    print("best_x_decoded", decode_solution(x_opt))
    # print("sigma:", sigma_opt)
    print("f :", f_opt)
    nas_ioh.f.reset() # Note that you must run the code after each independent run.



# If you are passing command line flags to modify the default config values, you
# must use app.run(main)
if __name__ == '__main__':
  start = time.time()
  app.run(main)
  end = time.time()
  print("The program takes %s seconds" % (end-start))
