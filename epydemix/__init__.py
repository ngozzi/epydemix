# epydemix/__init__.py

from .model.epimodel import EpiModel, simulate

__all__ = [
    'EpiModel',
    'simulate'
]
