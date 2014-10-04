from denoising_autoencoder_cart_pole import Cart_Pole_DA
import cart_pole
import cPickle

with open("cartDA.pickle") as f:
	agent = cPickle.load(f)
	cart = cart_pole.Cart(0.5)

	import PIL
	import lib.dlt_utils as utils
	import random
	image = PIL.Image.fromarray(utils.tile_raster_images(
		X=agent.weights.get_value(borrow=True).T,
		img_shape=(5, 4), tile_shape=(20, 20),
		tile_spacing=(1, 1)))
	image.save('../plots/pickled_cart_pole.png')

	def get_action(cart):
		return cart_pole.get_action(agent, cart)

	cart.reset()
	cart_pole.animate_cart(cart, get_action)
