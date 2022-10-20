"""Try to import all available dictionaries."""
import warnings


try:
    from ._dictionary_spams import DictionarySpams
except ImportError:
    warnings.warn("spams-python not found, 'DictionarySpams' won't be available", ImportWarning)

# try:
#     from ._dictionary_sklearn import DictionarySklearn
# except ImportError:
#     warnings.warn("scikit-learn not installed, 'DictionarySklearn' won't be available", ImportWarning)
