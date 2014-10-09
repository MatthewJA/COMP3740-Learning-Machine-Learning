import math
import cPickle

from mdp_da import MDP_DA
import cart_pole

with open("cartDA.pickle") as f:
	agent = cPickle.load(f)
	cart = cart_pole.Cart(math.pi/2-0.1)

	import PIL
	import lib.dlt_utils as utils
	import random
	image = PIL.Image.fromarray(utils.tile_raster_images(
		X=agent.weights.get_value(borrow=True).T,
		img_shape=(25, 25), tile_shape=(10, 20),
		tile_spacing=(1, 1)))
	image.save('../plots/pickled_cart_pole.png')

	def get_action(cart):
		return cart_pole.get_action(agent, cart)

	cart.reset()
	cart_pole.animate_cart(cart, get_action)
