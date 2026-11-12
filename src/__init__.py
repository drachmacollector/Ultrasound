"""Anatomy or Acquisition? study code.

Importing any ``src.*`` module first runs this file, which disables albumentations' import-time
version check (it makes a network request; the study pins albumentations==2.0.8).
"""
import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
