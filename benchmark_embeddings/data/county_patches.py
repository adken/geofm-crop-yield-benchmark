"""Raw Sentinel-2 county-patch access for supervised county regression.

This module follows the same two NPZ layouts used by the frozen-embedding
extractors and remains self-contained.
"""

from __future__ import annotations

import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from .io import (
    S2_10_BANDS,
    band_names,
    guess_input_bands,
    iter_patch_file_groups,
    load_yield_lookup,
    normalise_band_name,
    normalise_county,
    normalise_undersize_policy,
    npz_spatial_shape,
    npz_value,
    pixel_array,
    reindex_bands,
    safe_float,
    yield_bu_per_acre,
)


@dataclass(frozen=True)
class CountyPatchRecord:
    """Index metadata for one county-year without eagerly loading its pixels."""

    key: str
    county_id: str
    year: int
    target_bu_per_acre: float
    patch_ids: tuple[str, ...]
    shape: tuple[int, int, int, int, int]
    layout: str
    source: Any
    intervals: tuple[int, ...]
    input_bands: tuple[str, ...]
    source_spatial_shape: tuple[int, int] | None = None

    @property
    def num_patches(self) -> int:
        return self.shape[0]


def _npz_array_header(path: Path, key: str) -> tuple[tuple[int, ...], np.dtype]:
    """Read an array shape/dtype from an NPZ member without inflating it."""
    member = f"{key}.npy"
    with zipfile.ZipFile(path) as archive:
        if member not in archive.namelist():
            raise KeyError(f"{path} has no {key!r} array")
        with archive.open(member) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version in {(2, 0), (3, 0)}:
                shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
            else:  # pragma: no cover - NumPy currently emits only these versions
                raise ValueError(f"unsupported NPY header version {version} in {path}")
    return tuple(int(v) for v in shape), np.dtype(dtype)


def _normalise_bands(names: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        normalise_band_name(
            name.decode("utf-8") if isinstance(name, bytes) else str(name)
        )
        for name in names
    )


def _read_scalar(z, key: str, default: Any) -> Any:
    value = npz_value(z, key, default)
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _spatial_pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, (int, np.integer)):
        pair = (int(value), int(value))
    else:
        pair = tuple(int(v) for v in value)
        if len(pair) != 2:
            raise ValueError(
                "expected_spatial_size must be an integer or [height, width]"
            )
    if min(pair) <= 0:
        raise ValueError("expected_spatial_size values must be positive")
    return pair


class CountyPatchStore:
    """Index and lazily read raw Sentinel-2 county-year patch sequences.

    Supported layouts are the repository's bundled ``patches[P,T,C,H,W]``
    files and the per-patch/per-interval ``pixels[C,H,W]`` files.  Every
    returned tensor uses ``[P,T,C,H,W]`` and the canonical ten-band order.
    The production contract is 256x256: oversized inputs are deterministically
    center-cropped and undersized inputs are rejected.
    """

    def __init__(
        self,
        npz_dir: str | Path,
        *,
        yield_csv: str | Path | None = None,
        expected_timesteps: int = 7,
        expected_spatial_size: int | Sequence[int] = 256,
        oversize_policy: str = "center_crop",
        undersize_policy: str = "error",
        input_bands: Sequence[str] | None = None,
        require_complete_schedule: bool = True,
        fast_filename_index: bool = False,
        max_counties: int | None = None,
        io_workers: int | None = None,
    ):
        # Reading a county means opening one NPZ per patch per interval -- 56
        # files for the default eight patches over seven composites. Done
        # serially on the training thread that is the whole epoch: the GPU
        # waits on zlib. Decompression releases the GIL, so a thread pool
        # overlaps the reads. Order is preserved, so results are unchanged.
        if io_workers is None:
            io_workers = int(os.environ.get("BENCHMARK_IO_WORKERS", "8"))
        self.io_workers = max(1, int(io_workers))
        # One pool for the lifetime of the store. The first version built a
        # ThreadPoolExecutor inside load_patches, which runs once per chunk per
        # county -- thousands of create/destroy cycles per epoch, each spawning
        # and joining io_workers threads. That is how a job dies with SIGBUS
        # after forty minutes rather than failing immediately.
        self._pool = None
        self.npz_dir = Path(npz_dir)
        if not self.npz_dir.exists():
            raise FileNotFoundError(f"county NPZ directory does not exist: {self.npz_dir}")
        self.expected_timesteps = int(expected_timesteps)
        if self.expected_timesteps <= 0:
            raise ValueError("expected_timesteps must be positive")
        self.expected_spatial_size = _spatial_pair(expected_spatial_size)
        self.oversize_policy = str(oversize_policy).strip().lower()
        if self.oversize_policy not in {"center_crop", "error"}:
            raise ValueError("oversize_policy must be 'center_crop' or 'error'")
        self.undersize_policy = normalise_undersize_policy(undersize_policy)
        self.undersized_patches_excluded = 0
        self.undersized_county_years_excluded = 0
        self.target_bands = tuple(S2_10_BANDS)
        self.input_bands_override = (
            _normalise_bands(input_bands) if input_bands is not None else None
        )
        files = sorted(self.npz_dir.rglob("*.npz"))
        if not files:
            raise FileNotFoundError(f"no NPZ files found below {self.npz_dir}")
        self.yield_csv = Path(yield_csv).resolve() if yield_csv else None
        yield_lookup = load_yield_lookup(self.yield_csv) if self.yield_csv else {}
        with np.load(files[0], allow_pickle=True) as first:
            bundled = "patches" in first.files
        if bundled:
            records = self._index_bundled(
                files,
                yield_lookup=yield_lookup,
                require_complete_schedule=require_complete_schedule,
                max_counties=max_counties,
            )
        else:
            records = self._index_patch_files(
                files,
                yield_lookup=yield_lookup,
                require_complete_schedule=require_complete_schedule,
                fast_filename_index=fast_filename_index,
                max_counties=max_counties,
            )
        if not records:
            raise ValueError(
                "no labelled corn county-years remained after indexing; check labels, "
                "schedule length, and NPZ layout"
            )
        duplicate_keys = [key for key in {r.key for r in records} if sum(x.key == key for x in records) > 1]
        if duplicate_keys:
            raise ValueError(f"duplicate county-year records: {duplicate_keys[:5]}")
        self.records = sorted(records, key=lambda record: record.key)
        self.by_key = {record.key: record for record in self.records}

    def _index_bundled(
        self,
        files: Sequence[Path],
        *,
        yield_lookup: dict[tuple[str, int], float],
        require_complete_schedule: bool,
        max_counties: int | None,
    ) -> list[CountyPatchRecord]:
        records: list[CountyPatchRecord] = []
        for path in files:
            shape, _ = _npz_array_header(path, "patches")
            if len(shape) != 5:
                raise ValueError(f"{path}: expected patches [P,T,C,H,W], got {shape}")
            patches, timesteps, channels, height, width = shape
            if patches <= 0 or height <= 0 or width <= 0:
                raise ValueError(f"{path}: invalid patch shape {shape}")
            self._validate_spatial_size(height, width, context=str(path))
            if timesteps != self.expected_timesteps:
                if require_complete_schedule:
                    continue
            with np.load(path, allow_pickle=True) as z:
                county = normalise_county(_read_scalar(z, "county_fips", ""))
                year = int(safe_float(_read_scalar(z, "year", 0), 0))
                crop = str(_read_scalar(z, "crop", "corn")).strip().lower()
                embedded_yield = _read_scalar(
                    z,
                    "yield_bu_per_acre",
                    _read_scalar(z, "yield", np.nan),
                )
                yield_bu = safe_float(
                    yield_lookup.get(
                        (county, year),
                        np.nan if self.yield_csv else embedded_yield,
                    )
                )
                if self.input_bands_override is not None:
                    bands = self.input_bands_override
                elif "band_names" in z.files:
                    bands = _normalise_bands(z["band_names"].tolist())
                else:
                    bands = guess_input_bands(channels)
            if not county or year <= 0 or crop not in {"corn", "maize"}:
                continue
            if not np.isfinite(yield_bu):
                continue
            if len(bands) != channels:
                raise ValueError(
                    f"{path}: {channels} channels but {len(bands)} input band names"
                )
            key = f"{county}-{year}"
            records.append(
                CountyPatchRecord(
                    key=key,
                    county_id=county,
                    year=year,
                    target_bu_per_acre=float(yield_bu),
                    patch_ids=tuple(f"{key}-p{i}" for i in range(patches)),
                    shape=(
                        patches,
                        timesteps,
                        len(self.target_bands),
                        *self.expected_spatial_size,
                    ),
                    layout="bundled",
                    source=path,
                    intervals=tuple(range(timesteps)),
                    input_bands=tuple(bands),
                    source_spatial_shape=(height, width),
                )
            )
            if max_counties is not None and len(records) >= int(max_counties):
                break
        return records

    def _drop_undersized_entries(self, patch_entries: tuple) -> tuple:
        """Remove whole spatial patches that fall below the expected footprint.

        Drops the patch across every interval rather than individual files, so
        the retained sequences stay rectangular. Header-only reads, so this
        costs milliseconds per county-year.
        """
        target_height, target_width = self.expected_spatial_size
        kept = []
        for patch_id, by_interval in patch_entries:
            undersized = False
            for path in by_interval.values():
                try:
                    height, width = npz_spatial_shape(Path(path))
                except (KeyError, ValueError, OSError):
                    continue
                if height < target_height or width < target_width:
                    undersized = True
                    break
            if undersized:
                self.undersized_patches_excluded += 1
            else:
                kept.append((patch_id, by_interval))
        return tuple(kept)

    def _index_patch_files(
        self,
        files: Sequence[Path],
        *,
        yield_lookup: dict[tuple[str, int], float],
        require_complete_schedule: bool,
        fast_filename_index: bool,
        max_counties: int | None,
    ) -> list[CountyPatchRecord]:
        groups = iter_patch_file_groups(
            list(files),
            max_counties,
            yield_lookup,
            expected_timesteps=self.expected_timesteps,
            require_complete_schedule=require_complete_schedule,
            fast_filename_index=fast_filename_index,
            yield_lookup_is_authoritative=self.yield_csv is not None,
        )
        records: list[CountyPatchRecord] = []
        for group in groups:
            crop = str(group.get("crop", "corn")).strip().lower()
            if crop not in {"corn", "maize"}:
                continue
            yield_bu = yield_bu_per_acre(group)
            if not np.isfinite(yield_bu):
                continue
            patch_entries = tuple(group["patch_entries"])
            if not patch_entries:
                continue
            if self.undersize_policy == "skip":
                patch_entries = self._drop_undersized_entries(patch_entries)
                if not patch_entries:
                    # Every patch in this county-year was undersized.
                    self.undersized_county_years_excluded += 1
                    continue
            _, first_by_interval = patch_entries[0]
            first_path = first_by_interval[sorted(first_by_interval)[0]]
            with np.load(first_path, allow_pickle=True) as z:
                first = pixel_array(z)
                detected_bands = band_names(z)
            self._validate_spatial_size(
                int(first.shape[-2]),
                int(first.shape[-1]),
                context=str(first_path),
            )
            if self.input_bands_override is not None:
                bands = self.input_bands_override
            elif detected_bands is not None:
                bands = _normalise_bands(detected_bands)
            else:
                bands = guess_input_bands(first.shape[0])
            county = normalise_county(group["county_fips"])
            year = int(group["year"])
            intervals = tuple(int(v) for v in group["intervals"])
            records.append(
                CountyPatchRecord(
                    key=f"{county}-{year}",
                    county_id=county,
                    year=year,
                    target_bu_per_acre=float(yield_bu),
                    patch_ids=tuple(str(entry[0]) for entry in patch_entries),
                    shape=(
                        len(patch_entries),
                        len(intervals),
                        len(self.target_bands),
                        *self.expected_spatial_size,
                    ),
                    layout="patch_files",
                    source=patch_entries,
                    intervals=intervals,
                    input_bands=tuple(bands),
                    source_spatial_shape=(int(first.shape[-2]), int(first.shape[-1])),
                )
            )
        return records

    def subset(self, keys: Iterable[str]) -> list[CountyPatchRecord]:
        requested = [str(key) for key in keys]
        missing = sorted(set(requested).difference(self.by_key))
        if missing:
            raise ValueError(
                f"split contains {len(missing)} county-years absent from raw T{self.expected_timesteps} data: "
                f"{missing[:5]}"
            )
        return [self.by_key[key] for key in requested]

    def load_patches(
        self,
        record: CountyPatchRecord,
        indices: Sequence[int] | np.ndarray | None = None,
    ) -> torch.Tensor:
        selected = (
            np.arange(record.num_patches, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        if selected.ndim != 1 or selected.size == 0:
            raise ValueError("at least one patch index is required")
        if selected.min() < 0 or selected.max() >= record.num_patches:
            raise IndexError("patch index is outside the county record")
        if record.layout == "bundled":
            with np.load(record.source, allow_pickle=True) as z:
                array = z["patches"].astype(np.float32)[selected]
            array = self._scale_reflectance(array)
            array = np.stack(
                [
                    reindex_bands(
                        patch,
                        record.input_bands,
                        self.target_bands,
                    )
                    for patch in array
                ],
                axis=0,
            )
        else:
            if self.io_workers > 1 and len(selected) > 1:
                cubes = self._load_patch_cubes_concurrently(record, selected)
            else:
                cubes = [self._load_patch_file_cube(record, int(index)) for index in selected]
            shapes = {cube.shape for cube in cubes}
            if len(shapes) != 1:
                raise ValueError(
                    f"county {record.key} contains inconsistent patch shapes: {sorted(shapes)}"
                )
            array = np.stack(cubes, axis=0)
        array = self._conform_spatial_size(array, context=record.key)
        if array.shape[1] != self.expected_timesteps:
            raise ValueError(
                f"{record.key}: expected T={self.expected_timesteps}, got {array.shape}"
            )
        if array.shape[2] != len(self.target_bands):
            raise ValueError(f"{record.key}: expected ten S2 bands, got {array.shape}")
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
        return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))

    def _reader_pool(self) -> ThreadPoolExecutor:
        """Lazily create the shared reader pool, once per store."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self.io_workers, thread_name_prefix="patch-reader"
            )
        return self._pool

    def __del__(self):
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.shutdown(wait=False)

    def _load_patch_cubes_concurrently(
        self, record: CountyPatchRecord, selected: np.ndarray
    ) -> list[np.ndarray]:
        """Read every (patch, interval) file of this chunk in one parallel map.

        Parallelising over patches alone caps concurrency at the chunk size,
        because each patch still reads its seven composite files sequentially.
        Measured on the target filesystem, a single reader sees about 25 ms per
        file while 32 concurrent readers see 5.9 ms -- at which point the cost
        is zlib decompression rather than the filesystem. Flattening to one
        task per file is what makes that concurrency reachable: a county chunk
        of eight patches issues 56 reads at once instead of eight runs of seven.
        """
        tasks = [
            (position, int(interval), path)
            for position, index in enumerate(selected)
            for interval, path in record.source[int(index)][1].items()
        ]

        def read(task):
            position, interval, path = task
            return position, interval, self._read_interval_frame(
                record, path, interval
            )

        frames: list[dict[int, np.ndarray]] = [{} for _ in selected]
        for position, interval, frame in self._reader_pool().map(read, tasks):
            frames[position][interval] = frame
        return [
            self._assemble_patch_sequence(record, loaded, int(index))
            for loaded, index in zip(frames, selected)
        ]

    def _read_interval_frame(
        self, record: CountyPatchRecord, path: Any, interval: int
    ) -> np.ndarray:
        with np.load(path, allow_pickle=True) as z:
            frame = pixel_array(z)
            detected = band_names(z)
        bands = _normalise_bands(detected) if detected is not None else record.input_bands
        frame_t = reindex_bands(frame[None, ...], bands, self.target_bands)[0]
        return self._conform_spatial_size(
            frame_t, context=f"{path} (county {record.key}, interval {interval})"
        )

    def _assemble_patch_sequence(
        self,
        record: CountyPatchRecord,
        loaded: dict[int, np.ndarray],
        patch_index: int,
    ) -> np.ndarray:
        """Order the loaded frames onto the schedule, holding the last forward."""
        if not loaded:
            raise ValueError(f"{record.key} patch {patch_index} has no readable intervals")
        last = loaded[sorted(loaded)[0]]
        frames = []
        for interval in record.intervals:
            if interval in loaded:
                last = loaded[interval]
            frames.append(last)
        return np.stack(frames, axis=0).astype(np.float32)

    def _load_patch_file_cube(self, record: CountyPatchRecord, patch_index: int) -> np.ndarray:
        _, by_interval = record.source[patch_index]
        loaded: dict[int, np.ndarray] = {}
        for interval, path in by_interval.items():
            with np.load(path, allow_pickle=True) as z:
                frame = pixel_array(z)
                detected = band_names(z)
            bands = _normalise_bands(detected) if detected is not None else record.input_bands
            frame_t = reindex_bands(
                frame[None, ...], bands, self.target_bands
            )[0]
            frame_t = self._conform_spatial_size(
                frame_t,
                context=f"{path} (county {record.key}, patch {patch_index})",
            )
            loaded[int(interval)] = frame_t
        if not loaded:
            raise ValueError(f"{record.key} patch {patch_index} has no readable intervals")
        first = loaded[sorted(loaded)[0]]
        last = first
        frames = []
        for interval in record.intervals:
            if interval in loaded:
                last = loaded[interval]
            frames.append(last)
        return np.stack(frames, axis=0).astype(np.float32)

    def _validate_spatial_size(self, height: int, width: int, *, context: str) -> None:
        expected_height, expected_width = self.expected_spatial_size
        if height < expected_height or width < expected_width:
            raise ValueError(
                f"{context}: source patch is {height}x{width}, below required "
                f"{expected_height}x{expected_width}; padding or upsampling is not allowed"
            )
        if (
            (height, width) != self.expected_spatial_size
            and self.oversize_policy == "error"
        ):
            raise ValueError(
                f"{context}: source patch is {height}x{width}, expected exactly "
                f"{expected_height}x{expected_width}"
            )

    def _conform_spatial_size(self, array: np.ndarray, *, context: str) -> np.ndarray:
        height, width = (int(array.shape[-2]), int(array.shape[-1]))
        self._validate_spatial_size(height, width, context=context)
        expected_height, expected_width = self.expected_spatial_size
        if (height, width) == self.expected_spatial_size:
            return array
        top = (height - expected_height) // 2
        left = (width - expected_width) // 2
        return array[
            ...,
            top : top + expected_height,
            left : left + expected_width,
        ]

    @staticmethod
    def _scale_reflectance(array: np.ndarray) -> np.ndarray:
        finite = array[np.isfinite(array)]
        if finite.size and float(finite.max()) > 2.0:
            array = array / 10000.0
        return np.clip(array, 0.0, 1.0).astype(np.float32)

    def describe(self) -> dict[str, Any]:
        model_shapes = sorted({record.shape[1:] for record in self.records})
        source_shapes = sorted(
            {
                (
                    record.shape[1],
                    record.shape[2],
                    *(record.source_spatial_shape or record.shape[-2:]),
                )
                for record in self.records
            }
        )
        counts = np.array([record.num_patches for record in self.records], dtype=np.int64)
        crop_count = sum(
            (record.source_spatial_shape or record.shape[-2:])
            != self.expected_spatial_size
            for record in self.records
        )
        expected_height, expected_width = self.expected_spatial_size
        return {
            "num_county_years": len(self.records),
            "yield_labels": str(self.yield_csv) if self.yield_csv else "embedded_fallback",
            "target_units": "bushels_per_acre",
            "source_input_contract": "[num_patches,time,channels,height,width]",
            "input_contract": (
                f"[num_patches,{self.expected_timesteps},{len(self.target_bands)},"
                f"{expected_height},{expected_width}]"
            ),
            "expected_spatial_size": list(self.expected_spatial_size),
            "oversize_policy": self.oversize_policy,
            "undersize_policy": self.undersize_policy,
            "undersized_spatial_patches_excluded": int(
                self.undersized_patches_excluded
            ),
            "undersized_county_years_excluded": int(
                self.undersized_county_years_excluded
            ),
            "crop_anchor": "center",
            "indexed_source_per_patch_shapes": [list(shape) for shape in source_shapes],
            "observed_per_patch_shapes": [list(shape) for shape in model_shapes],
            "indexed_county_years_requiring_crop": int(crop_count),
            "bands": list(self.target_bands),
            "patch_count_min": int(counts.min()),
            "patch_count_median": float(np.median(counts)),
            "patch_count_max": int(counts.max()),
        }


@dataclass(frozen=True)
class BandNormalizer:
    mean: tuple[float, ...]
    std: tuple[float, ...]

    def transform(self, patches: torch.Tensor) -> torch.Tensor:
        mean = patches.new_tensor(self.mean).view(1, 1, -1, 1, 1)
        std = patches.new_tensor(self.std).view(1, 1, -1, 1, 1)
        return (patches - mean) / std

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": list(self.mean), "std": list(self.std)}


@dataclass(frozen=True)
class TargetScaler:
    mean: float
    std: float

    @classmethod
    def fit(cls, records: Sequence[CountyPatchRecord]) -> "TargetScaler":
        targets = np.array(
            [record.target_bu_per_acre for record in records], dtype=np.float64
        )
        if targets.size == 0 or not np.isfinite(targets).all():
            raise ValueError("target scaling requires finite training targets")
        std = float(targets.std(ddof=0))
        return cls(mean=float(targets.mean()), std=std if std > 1e-8 else 1.0)

    def transform_tensor(self, target: torch.Tensor) -> torch.Tensor:
        return (target - self.mean) / self.std

    def inverse_tensor(self, target: torch.Tensor) -> torch.Tensor:
        return target * self.std + self.mean

    def inverse_array(self, target: np.ndarray) -> np.ndarray:
        return np.asarray(target) * self.std + self.mean

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}


def fit_band_normalizer(
    store: CountyPatchStore,
    records: Sequence[CountyPatchRecord],
    *,
    seed: int,
    max_patches_per_county: int | None = None,
    max_pixels_per_patch: int | None = None,
    chunk_size: int = 4,
) -> BandNormalizer:
    """Fit streaming band statistics using training county-years only."""
    channels = len(store.target_bands)
    sums = np.zeros(channels, dtype=np.float64)
    squared = np.zeros(channels, dtype=np.float64)
    counts = np.zeros(channels, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    for record in records:
        indices = np.arange(record.num_patches, dtype=np.int64)
        if max_patches_per_county and len(indices) > int(max_patches_per_county):
            indices = np.sort(
                rng.choice(indices, size=int(max_patches_per_county), replace=False)
            )
        for start in range(0, len(indices), max(1, int(chunk_size))):
            patches = store.load_patches(record, indices[start : start + chunk_size]).numpy()
            if max_pixels_per_patch and patches.shape[-2] * patches.shape[-1] > int(max_pixels_per_patch):
                spatial = patches.shape[-2] * patches.shape[-1]
                pick = np.sort(
                    rng.choice(spatial, size=int(max_pixels_per_patch), replace=False)
                )
                patches = patches.reshape(*patches.shape[:-2], spatial)[..., pick]
            values = np.moveaxis(patches, 2, 0).reshape(channels, -1).astype(np.float64)
            for channel in range(channels):
                finite = values[channel][np.isfinite(values[channel])]
                if finite.size:
                    sums[channel] += finite.sum()
                    squared[channel] += np.square(finite).sum()
                    counts[channel] += finite.size
    if np.any(counts == 0):
        raise ValueError(f"cannot fit band statistics; empty channels {np.where(counts == 0)[0]}")
    mean = sums / counts
    variance = np.maximum(squared / counts - np.square(mean), 0.0)
    std = np.sqrt(variance)
    std[std < 1e-6] = 1.0
    return BandNormalizer(
        mean=tuple(float(value) for value in mean),
        std=tuple(float(value) for value in std),
    )


def deterministic_patch_sample(
    record: CountyPatchRecord,
    patches_per_county: int | None,
    *,
    seed: int,
    epoch: int,
) -> np.ndarray:
    """Sample a reproducible, epoch-varying patch subset without replacement."""
    indices = np.arange(record.num_patches, dtype=np.int64)
    if not patches_per_county or len(indices) <= int(patches_per_county):
        return indices
    key_seed = sum((index + 1) * ord(char) for index, char in enumerate(record.key))
    rng = np.random.default_rng(int(seed) + 1_000_003 * int(epoch) + key_seed)
    return np.sort(rng.choice(indices, size=int(patches_per_county), replace=False))
