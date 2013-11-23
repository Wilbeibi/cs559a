from math import *
import numpy as np


def circ(r,theta,x0,y0):
	x = x0 + r * cos(theta)
	y = y0 + r * sin(theta)
	return (x,y)

def arc(x0,y0,theta_i,theta_f):
	r = 1
	thetas = np.arange(theta_i,theta_f,2.*pi/100)
	for theta in thetas:
		yield circ(r,theta,x0,y0)

def print_arc(x0,y0,theta_i,theta_f,label):
	for xy in arc(x0,y0,theta_i,theta_f):
		print "%1.3f,%1.3f,%d" % (xy[0],xy[1],label)

#0
print_arc(0,0,-pi/4,5./4*pi,0)
#1
print_arc(1,0,-pi/4-pi,5./4*pi-pi,1)
