"""Kalshi order executor. Separate from predlab on purpose: predlab never imports it.

Modes: dry_run (default, nothing sent), demo (Kalshi demo exchange, mock funds),
live (real money; requires configured limits AND an open go-live gate per model).
"""
