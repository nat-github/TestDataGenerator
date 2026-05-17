"""Post-generation data validators.

Currently contains the Great Expectations adapter that auto-derives
expectations from `models.config_models.ColumnConfig` and validates
generated DataFrames against them.

See `validators/gx_validator.py` and `Data_Validation.md`.
"""
