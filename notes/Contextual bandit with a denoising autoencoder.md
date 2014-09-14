Contextual bandit with a denoising autoencoder
==============================================

Cost becomes the difference between the expected reward and the actual reward.
We award a reward of 1 for every correct label and 0 otherwise.

To actually calculate this we will get a vector of predicted labels and check
equality on them, interpreting equality as a reward of 1 and inequality as a
reward of 0. We will then interpret the probability vector itself as expected
reward, and get the absolute value of the difference between it and the reward
vector. We will then average this to get the average cost.

This is implemented in denoising_autoencoder_contextual_bandit.py.