import numpy as np

from . import util


def _gen_parents_inplace(signal, dictionaries, sc_lambda, parents, **kwargs_dico):
	for ip, (key, dico) in enumerate(dictionaries.items()):
		parents[:,ip] = dico.reconstruct(signal, sc_lambda, **kwargs_dico)


def _gen_parents_auto_inplace(signal, dictionaries, parents, **kwargs_dico):
	lambdas = []
	for ip, (key, dico) in enumerate(dictionaries.items()):
		result = dico.reconstruct_auto(signal, full_output=True, **kwargs_dico)
		parents[:,ip] = result[0]
		lambdas.append(result[2]['x'])

	return lambdas


def gen_parents_batch(signals, dictionaries, sc_lambda, parents=None, normalize=False,
					  verbose=False, **kwargs_dico):
	"""TODO
	
	Genera els parents d'un conjunt de senyals en un array (longitud, senyals)
	de numpy amb els diccionaris de grawadile dins un dict. Si es dona el
	'parents' (l_window, n_dicos, n_signals) es clavaran els valors a dintre en
	comptes de crear un nou array per als resultats.

	"""
	l_window, n_signals = signals.shape
	n_dicts = len(dictionaries)

	if parents is None:
		parents = np.empty((l_window, n_dicts, n_signals), order='F')

	for isi in range(n_signals):
		if verbose:
			print(f"Generating parents of #{isi}...")
		_gen_parents_inplace(
			signals[:,isi],
			dictionaries,
			sc_lambda,
			parents[...,isi],
			**kwargs_dico
		)

	if normalize:
		util.abs_normalize(parents)

	return parents


def gen_parents_auto_batch(signals, dictionaries, parents=None, l2_normed=True, verbose=False,
						   **kwargs_dico):
	"""TODO
	
	Genera els parents d'un conjunt de senyals en un array (longitud, senyals)
	de numpy amb els diccionaris de grawadile dins un dict. La lambda de
	reconstrucció es calcula per bisecció tal que s'anul·le el màrge esquerre
	especificat als `kwargs_dico`.
	Si es dona el 'parents' (l_window, n_dicos, n_signals) es clavaran els 
	valors a dintre en comptes de crear un nou array per als resultats.

	"""
	l_window, n_signals = signals.shape
	n_dicts = len(dictionaries)

	if parents is None:
		parents = np.empty((l_window, n_dicts, n_signals), order='F')
	lambdas = np.empty((n_dicts, n_signals), order='F')

	for isi in range(n_signals):
		if verbose:
			print(f"Generating parents of #{isi}...")
		lambdas[:,isi] = _gen_parents_auto_inplace(
			signals[:,isi],
			dictionaries,
			parents[...,isi],
			**kwargs_dico
		)

	if l2_normed:
		util.l2_normalize(parents)

	return (parents, lambdas) if parents is None else lambdas
