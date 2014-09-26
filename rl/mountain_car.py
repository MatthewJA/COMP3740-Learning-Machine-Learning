
# We're on a quadratic slope: y = c x^2
# Gradient(x) = 2x, so acceleration = -2cgx

# action \in {-1, 0, 1}

class MountainCar(object):
  def __init__(self):
    self.c = 0.01
    self.g = 9.8
    self.accelerator = 3
    self.boundary = 100
    self.position = 0
    self.velocity = 0

  # returns (position, velocity, completed)
  def step(action):
    self.velocity += action * self.accelerator - 2 * self.g * self.position * self.c
    self.position += self.velocity
    return self.output_grid()

  def output_grid:
    output = []
    for position in range(-100, 100, 10):
      for velocity in range(-10, 10, 2.5):
        output.append(0.9**((self.position - position)**2 +
                            (self.velocity - velocity)**2))

    return output