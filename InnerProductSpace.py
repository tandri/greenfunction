
# Python 2.7.6
#
#	TG
#	11/08/2015
#

import numpy as np



# Default inner product

def dotProduct( u, v ):
	'''

	r = dotProduct( u, v )

	u,v 	complex 1d-arrays
			len(u) == len(v)


	r is the dot product of u and v in C^n.

		r = sum_j u[j] * conjugate( v[j] )


	'''


	v_conj = np.conjugate(v)

	return (u*v_conj).sum()




class InnerProductSpace:
	'''

	Inner product space, V, of finite dimension over C.

	V = InnerProductSpace( dim, inProd )

	dim 	int 	Dimension of V
	inProd 	func	The inner product of V is < u, v > := inProd( u, v )


	Vectors are represented by their coordinate vectors, complex np.ndarrays of shape (dim,).

	'''



	def __init__( self, dim, inProd = dotProduct ):
		'''

		V__init__( dim, inProd )

		dim 		integer
		inProd 		function (default: dotProduct)
					Takes as argument two 1d-arrays with equal length.
					Returns a real/complex number.

		V is an dim dimensional inner product space with inner product
			< u, v > := inProd( u, v )

		'''

		self.dim = int(dim)
		self.inProd = lambda u, v: self._safeInnerProduct( u, v, inProd )



	def _safeInnerProduct( self, u, v, inProd ):
		'''

		r = V._safeInnerProduct( u, v, inProd )

		u,v 		complex 1d-arrays
					len(u) == len(v)
		inProd 		function (default: dotProduct)
					Takes as argument two 1d-arrays with equal length.
					Returns a real/complex number.
		
		r = < u, v > = inProd( u, v )
					Makes sure that u and v are of correct size and type.

		'''


		if not self.contains( u ):
			u = self.cast( u )


		if not self.contains( v ):
			v = self.cast( v )


		return inProd( u, v )



	def contains( self, u ):
		'''

		b = V.contains( u )

		b == True iff type(u) is np.ndarray and len(u) == V.dim

		'''
		return type(u) is np.ndarray and u.shape == (self.dim,)



	def cast( self, u ):
		'''
		
		v = V.cast(u)

		u 		array-like

		v 		is u cast to complex np.ndarray

		Raises error if v.shape is not (V.dim,)

		'''

		v = np.array(u) + 0j

		v = v.flatten()

		if len(v) != self.dim:
			raise RuntimeError('Wrong vector dimension! Space dim: %d, Input dim: %d' % ( self.dim, len(v) ) )

		return v



	def norm( self, u ):
		'''

		r = V.norm( u )

		u 	 	complex 1d-array
				len(u) == V.dim

		r = || u || := sqrt( < u, u > )		

		'''

		if not self.contains( u ):
			u = self.cast( u )

		return np.sqrt( self.inProd( u, u ) ).real



	def dist( self, u, v ):
		'''

		d = V.dist( u, v )

		u,v	 	complex 1d-arrays
				len(u) == len(v) == V.dim

		d = || u - v || := sqrt( < u-v, u-v > )		

		'''


		if not self.contains( u ):
			u = self.cast( u )



		if not self.contains( v ):
			v = self.cast( v )


		w = u - v

		return np.sqrt( self.inProd( w, w ) ).real



	def proj( self, u, v ):
		'''

		w = V.proj( u, v )

		u,v 	complex 1d-arrays
				len(u) == len(v) == V.dim

		w 		Orthogonal projection of v onto the vector u with respect to the inner product < u, v >

		'''


		if not self.contains( u ):
			u = self.cast( u )


		if not self.contains( v ):
			v = self.cast( v )


		scale = self.inProd( u, u )


		if scale == 0:
			raise RuntimeError( 'Cannot project onto zero vector' )


		return self.inProd( v, u ) * u / scale



	def GramSchmidt( self ):
		'''

		B = V.GramSchmidt( A )

		B 		complex 2d-array
				B.shape == ( V.dim, V.dim )
				B[0], ..., B[n-1] is an orthonormal basis for V with respect to the inner product < u, v >
				obtained by the Gram Schmit process on the standard basis.

		'''

		# Old basis
		A = np.eye( self.dim )
		
		# New basis
		B = A + 0j

		for i in range( self.dim ):
			
			# We have found the first i elements of the
			# orthonormal basis.

			for j in range(i):
				B[i] -= self.proj( B[j], A[i] )

			B[i] = B[i] / self.norm( B[i] )

		return B



	def dimension( self ):
		return self.dim




def polyProduct( u, v ):
	'''

	r = inProd( u, v )

	u,v 	complex 1d-arrays

	r = < u, v > is the inner product of u and v defined in the following way:

		Define polynomials
				p(z) = sum_j  u[j] * z^j
			and q(z) = sum_j  v[j] * z^j

		r = < u, v > := integral_[0,1]  p * conjugate(q) dx

	'''

	dx = 0.01
	x = np.arange( 0, 1, dx )

	p = np.polynomial.polynomial.polyval( x, u )
	q = np.polynomial.polynomial.polyval( x, v )

	return ( p * np.conjugate( q ) * dx ).sum()




def main():

	V = InnerProductSpace( 3, polyProduct )
	B = V.GramSchmidt()

	print InnerProductSpace.__doc__




if __name__ == '__main__':
	main()