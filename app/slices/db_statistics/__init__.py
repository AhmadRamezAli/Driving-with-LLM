"""
Database Statistics Slice

This slice provides APIs for retrieving statistical information about the database,
focusing on model performance, scene characteristics, and the relationship between
ego-vehicle state and model predictions.
"""

from .routers import router

__all__ = ["router"] 