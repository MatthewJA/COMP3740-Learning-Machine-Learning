# contextual bandit with RMI

To make a contextual bandit agent with RMI, we need to change the cost function slightly. Our model will be a similar to a denoising autoencoder.

Suppose our observations are in $R^n$ and we have m actions. When the agent recieves an observation, we tack on $m$ zeroes to our observation, so now we have something in $R^{n+m}$. We run it through our denoising filter. Now, our last $m$ entries in the denoised vector are the predicted rewards for various choices of action which we can make.

For efficiency's sake, we can run a minibatch of a few hundred of these trials. Now we have a few hundred (observation, action, reward) tuples.

Now, we train our model with the cost function:

  Cost(observation, action, reward) = distance(observation, add_noise(observation)) + lambda * distance(reward, predicted_reward)