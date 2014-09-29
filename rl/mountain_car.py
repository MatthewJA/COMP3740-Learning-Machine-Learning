import numpy
import random
import pylab

# We're on a quadratic slope: y = c x^2
# Gradient(x) = 2x, so acceleration = -2cgx

# action \in {-1, 0, 1}


def get_output_function(variables):
  def closeness(vector1, vector2):
    return 0.9 ** ((x-y)**2 for (x,y) in zip(vector1, vector2))

  input_vector = list([x] for x in variables[0][1])
  for variable_name, variable_range in variables[1:]:
    input_vector = [x + [y] for x in input_vector for y in variable_range]

  def output_function(self):
    actual_position = [self.__dict__[name] for (name, _) in variables]
    return [closeness(actual_position, other_position)
              for other_position in input_vector]

  return output_function

class MountainCar(object):
  def __init__(self):
    self.c = 0.01
    self.g = 9.8
    self.accelerator = 0.5
    self.boundary = 100
    self.position = 20
    self.velocity = 0

  # returns (position, velocity, completed)
  def step(self,action):
    self.velocity += action * self.accelerator - 2 * self.g * self.position * self.c
    self.position += self.velocity
    return (self.position, self.velocity, abs(self.position) > 100)

  output = get_output_function([("position", range(-100,100,10)),
                                ("velocity", range(-10,10,10))])

m = MountainCar()

positions, velocities = [], []
velocity = 0

while True:
  action = 1 if velocity > 0 else -1
  position, velocity, done = m.step(action)
  positions.append(position)
  velocities.append(velocity)

  if done:
    break
  else:
    print position, velocity

print "Good job!"

pylab.plot(positions)

pylab.show()