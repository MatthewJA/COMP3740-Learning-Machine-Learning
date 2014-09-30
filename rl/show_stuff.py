import pylab
smoothing = 10
stuff = list(int(x) for x in open("output_2"))
print len(stuff)
pylab.plot([sum(stuff[i:i+smoothing])/len(stuff[i:i+smoothing])
                                for i in range(0, len(stuff), smoothing)])
pylab.show()