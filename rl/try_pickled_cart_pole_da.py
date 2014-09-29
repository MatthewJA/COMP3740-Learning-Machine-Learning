import cart_pole
import cPickle

with open("cartDA.pickle") as f:
	agent = cPickle.load(f)
	cart = cart_pole.Cart(0.1)

	def get_action(cart):
		return cart_pole.get_action(agent, cart)

	cart.reset()
	cart_pole.animate_cart(cart, get_action)