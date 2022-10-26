import numpy as np

from . import util


def _gen_parents_inplace(signal, dictionaries, sc_lambda, out, **kwargs):
	for ip, (key, dico) in enumerate(dictionaries.items()):
		out[:,ip] = dico.reconstruct(signal, sc_lambda, **kwargs)


def _gen_parents_inplace_auto(signal, dictionaries, out, **kwargs):
	for ip, (key, dico) in enumerate(dictionaries.items()):
		out[:,ip] = dico.reconstruct_auto(signal, **kwargs)


def gen_parents(signal, dictionaries, sc_lambda=None, out=None, **kwargs):
	"""TODO
	Genera els parents d'un senyal en un array numpy amb els diccionaris de
	grawadile dins un dict. Si es dona el 'out' es clavaran els valors a
	dintre en comptes de crear un nou array per als resultats. Els kwargs es
	passen directament a cada diccionari de grawadile.

	"""
	if out is None:
		l_window = len(signal)
		n_dicts = len(dictionaries)
		out = np.empty((l_window, n_dicts), order='F')
	
	if sc_lambda is None:
		_gen_parents_inplace(signal, dictionaries out, **kwargs)
	else:
		_gen_parents_inplace(signal, dictionaries, sc_lambda, out, **kwargs)

	return out


def gen_parents_batch(signals, dictionaries, sc_lambda=None, out=None, normalize=False, verbose=False, **kwargs):
	"""TODO
	Genera els parents d'un conjunt de senyals en un array (longitud, senyals)
	de numpy amb els diccionaris de grawadile dins un dict. Si es dona el
	'out' (l_window, n_dicos, n_signals) es clavaran els valors a dintre en
	comptes de crear un nou array per als resultats. Els kwargs es passen
	directament a cada diccionari de grawadile.

	"""
	l_window, n_signals = signals.shape
	n_dicts = len(dictionaries)

	if out is None:
		out = np.empty((l_window, n_dicts, n_signals), order='F')

	for isi in range(n_signals):
		if verbose:
			print(f"Generating parents of #{isi}...")
		if sc_lambda is None:
			_gen_parents_inplace_auto(
				signals[:,isi],
				dictionaries,
				out[...,isi],
				**kwargs
			)
		else:
			_gen_parents_inplace(
				signals[:,isi],
				dictionaries,
				sc_lambda,
				out[...,isi],
				**kwargs
			)

	if normalize:
		util.abs_normalize(out)

	return out
