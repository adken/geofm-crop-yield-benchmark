"""Adapters from historical GeoFM outputs to one benchmark schema."""

from importlib import import_module

from .adapters import (
    adapt_alphaearth_csv,
    adapt_embedding_parquet,
    adapt_npz_directory,
    adapt_presto_arrays,
)
from .schema import REQUIRED_COLUMNS, read_embeddings, validate_embeddings, write_embeddings

_CLAY_EXPORTS = {
    "ClayPatchDataset",
    "ClaySensorMetadata",
    "encode_clay_location",
    "encode_clay_time",
    "extract_clay_embeddings",
    "pool_clay_tokens",
}

_PRESTO_EXPORTS = {
    "PrestoPatchDataset",
    "build_presto_batch",
    "extract_presto_embeddings",
    "load_presto",
}

_TERRAMIND_EXPORTS = {
    "TerraMindPatchDataset",
    "extract_terramind_embeddings",
    "load_terramind",
    "pool_terramind_tokens",
    "prepare_terramind_bands",
    "terramind_model_kwargs",
}

_PRITHVI_EXPORTS = {
    "PrithviPatchDataset",
    "expected_token_count",
    "extract_prithvi_embeddings",
    "load_prithvi",
    "pool_prithvi_tokens",
}


def __getattr__(name):
    if name in _CLAY_EXPORTS:
        return getattr(import_module(".clay", __name__), name)
    if name in _PRESTO_EXPORTS:
        return getattr(import_module(".presto", __name__), name)
    if name in _TERRAMIND_EXPORTS:
        return getattr(import_module(".terramind", __name__), name)
    if name in _PRITHVI_EXPORTS:
        return getattr(import_module(".prithvi", __name__), name)
    raise AttributeError(name)

__all__ = [
    "REQUIRED_COLUMNS",
    "adapt_alphaearth_csv",
    "adapt_embedding_parquet",
    "adapt_npz_directory",
    "adapt_presto_arrays",
    "ClayPatchDataset",
    "ClaySensorMetadata",
    "encode_clay_location",
    "encode_clay_time",
    "extract_clay_embeddings",
    "pool_clay_tokens",
    "PrestoPatchDataset",
    "build_presto_batch",
    "extract_presto_embeddings",
    "load_presto",
    "TerraMindPatchDataset",
    "extract_terramind_embeddings",
    "load_terramind",
    "pool_terramind_tokens",
    "prepare_terramind_bands",
    "terramind_model_kwargs",
    "PrithviPatchDataset",
    "expected_token_count",
    "extract_prithvi_embeddings",
    "load_prithvi",
    "pool_prithvi_tokens",
    "read_embeddings",
    "validate_embeddings",
    "write_embeddings",
]
