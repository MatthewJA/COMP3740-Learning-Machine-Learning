
# We're on a quadratic slope: y = c x^2
# Gradient = 2x, so acceleration = -2cgx

# action \in {-1, 0, 1}

c = 0.01
g = 9.8
accelerator = 3
boundary = 100

# returns (position, velocity, completed)
def mountain_car_step(position, velocity, action):
  velocity += action * accelerator - 2 * g * x
  position += velocity
  return (position, velocity, abs(position) > boundary)

