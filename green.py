

# Python 2.7.6
#
#	TG
#	11/08/2015
#


import numpy as np

import InnerProductSpace

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm




def regionPoints( corner = 0j, width = 1., condition = lambda z: 0*z + 1, N = 100 ):
	'''

	Points, weight = regionPoints( corner, width, condition )

	corner		complex number 		(default: 0)
	width		real number 		(default: 1)
	condition	function that takes in an array of complex numbers and returns 0/False or 1/True
									(default: constant function 1)
	N 			integer 			(default 100)

	We identify K as the region inside the square S:

	  (corner + i*width) ------------- (corner + (1+i)*width)
						 |    __   S |
						 |   /	\    |
						 |  / K  )   |
						 | |	(    |
						 |  \___/    |
				(corner) ------------- (corner + width)

	that fulfills condition. That is
			A point z is in K if and only if z is in S and condition(z) == 1/True


	Points 		complex 1d-array of the numbers in K (at most all N^2 points of S).
	weight 		real, the area each point represents.


	'''

	
	minRe = corner.real
	maxRe = minRe + width
	minIm = corner.imag
	maxIm = minIm + width

	xx = np.linspace( minRe, maxRe, N + 1 )
	yy = np.linspace( minIm, maxIm, N + 1 )
	Re, Im = np.meshgrid( xx, yy )
	Grid = Re + Im*1j

	Points = Grid[ condition( Grid ).astype( bool ) ]

	weight = (width / N)**2

	return Points, weight



def innerProduct( u, v, n, Q, K ):
	'''

	r = inProd( u, v, n, Q, K )

	n 		integer
	u,v 	complex 1d-arrays (representing polynomials)
			len(u) == len(v) == n
	Q 		weight function
	K = ( Points, weight )
			Points 		complex array
			weight 		real

	r = <u,v>
		where <u,v> is an inner product on a polynomial space defined in the following way:

		We interpret K to be a region containing the numbers 'Points', each one representing the area 'weight'.

		Define polynomials
				p(z) = sum_j  u[j] * z^j
			and q(z) = sum_j  v[j] * z^j

		<u,v> := integral_K  p * conjugate(q) * exp( -2*n*Q ) dm

	'''

	Points, weight = K

	p = np.polynomial.polynomial.polyval( Points, u )
	q = np.polynomial.polynomial.polyval( Points, v )

	Int = weight * ( p * np.conjugate(q) * np.exp( -2.*n * Q( Points ) ) )
	
	return Int.sum()



def Bergman( z, B ):
	'''

	S = Bergman( z, B )


	z 		complex array (or number)
	B 		n*n complex 2d-array
			B[j] coefficients of a polynomial p_j


	S = sum_j  |p_j(z)|^2

	'''

	S = 0

	for j in range( len(B) ):

		p_j = np.polynomial.polynomial.polyval( z, B[j] )
		
		S += p_j*np.conjugate(p_j)

	# S has no imaginary part. Cast to real.
	return S.real



def Green( z, n, Q = lambda z: 0*z, K = regionPoints() ):
	'''

	g = Green( z, n, Q, K )


	z 		complex array (or number)
	n 		integer
	Q 		weight function				(default: constant function 0)
	K = ( Points, weight )
			Points 		complex array 	(default: unit square [0,1] + [0,1]i )
			weight 		real			(default: 10^-4)


	g 		The n-th approximation of the weighted Green function
			G_K_Q(z) = sup{ u(z) : u in L(C), u <= Q on K }
			evaluated at z.


	'''

	inProd = lambda u,v: innerProduct( u, v, n, Q, K )

	V = InnerProductSpace.InnerProductSpace( n, inProd )

	B = V.GramSchmidt()
	
	return np.log( Bergman( z, B ) ) / (2.*n)



def drawGreen( n, Q = lambda z: 0*z, K = regionPoints(), show = True ):
	'''

	fig = drawGreen( n, Q, K )


	n 		integer
	Q 		weight function					(default: constant function 0)
	K = ( Points, weight )
			Points 		complex array 		(default: unit square [0,1] + [0,1]i )
			weight 		real				(default: 10^-4)
	show 	boolean (default: True)


	
	fig 	Shows the n-th approximation of the weighted Green function
				G_K_Q(z) = sup{ u(z) : u in L(C), u <= Q on K }
			in a neighborhood of K.
			If show == True, plt.show() is called.


	'''

	Points = K[0]

	minRe = Points.real.min()
	maxRe = Points.real.max()
	minIm = Points.imag.min()
	maxIm = Points.imag.max()

	reMargin = (maxRe - minRe) * 1.5
	imMargin = (maxIm - minIm) * 1.5

	minRe -= reMargin
	maxRe += reMargin
	minIm -= imMargin
	maxIm += imMargin

	N = 100

	xx = np.linspace(minRe,maxRe,N+1)
	yy = np.linspace(minIm,maxIm,N+1)
	Re, Im = np.meshgrid(xx,yy)
	Z = Re + Im*1j
	
	green = Green( Z, n, Q, K )

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	surf = ax.plot_surface(Re, Im, green, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

	if show:
		plt.show()

	return fig



def GreenEll( z, a, b ):
	'''
	Green fall med dK = sporbaugur og Q = 0.
	GreenEll(z) = max{ 0, log( sqrt( (x/a)^2 + (y/b)^2 ) ) }
	'''

	r = (z.real/a)**2 + (z.imag/b)**2
	gr = np.log( r*(r>1) + 1.*(r<=1) ) / 2.
	return gr



def main():

	# Max degree of polynomials
	n = 100

	# Lower left corner and width of square S
	corner = - 5 - 5j
	width = 10.

	# z is in K iff z is in S and condition(z) == 1/True
	a = 1.
	b = 1.5
	condition = lambda z: (z.real/a)**2 + (z.imag/b)**2 < 1
	
	K = regionPoints( corner, width, condition, 100 )

	Q = lambda z: 0.*z
	
	drawGreen( n, Q, K, show = False )


	xx = np.linspace(-2,2,51)
	yy = np.linspace(-2,2,51)
	Re, Im = np.meshgrid(xx,yy)
	Z = Re + Im*1j
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	surf = ax.plot_surface(Re, Im, GreenEll(Z,a,b), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	plt.show()



if __name__ == '__main__':
    main()
