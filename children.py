import numpy as np
import sklearn


def omp_single_match(signal, dataset, **kwargs):
	"""TODO
	Açò és un cacau. Revisar la documentació de 
	`sklearn.decomposition.sparse_encode`, que és més adequat al que necessite
	que el SparseCoder. Valorar quin em convé, aquest el puc aplicar en
	qualsevol moment, però el SparseCoder convé inicialitzar-lo en el nivell
	dels loops on només s'inicialitze una vegada per dataset.
	I nano:
		- Gasta l'habilitat d'ambdós per processar múltiples senyals en comptes
		d'iterar a ma sobre cadascuna, molt més eficient per paral·lelitzar.
		- Al cas de només nonzeros=1 no fer el producte matricial amb el
		diccionari, és redundant i innecessari. Serà suficient amb guardar
		el `code` (i pot ser tornar l'àtom sel·leccionat).

	"""
	pass