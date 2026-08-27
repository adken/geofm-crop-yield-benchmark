#!/usr/bin/env python3
r"""Plot Sentinel-2 patch-sequence centroids over county boundaries.

The benchmark stores one NPZ file per patch and 28-day composite. This script
groups those files by county, year, and projected ``x/y`` patch identifier,
keeps complete seven-composite sequences by default, and reads ``raw_lat`` and
``raw_lon`` from one representative NPZ in each sequence. The points are then
drawn over a county-boundary GeoJSON without requiring GeoPandas or Shapely.

Run from the benchmark repository root::

    python scripts/plot_patch_locations.py \
        --npz-dir data/patches/sentinel-2-l2a \
        --geometry data/geometry/corn_boundary.geojson \
        --cohort outputs/cohort_covered/group_kfold_county_tabular.csv \
        --out figures/sentinel2_patch_locations.pdf

The output is accompanied by ``.points.csv`` and ``.contract.json`` sidecars.
Use ``--max-points`` for a quick preview; omit it for the publication figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PATCH_RE = re.compile(
    r"county_(?P<county>\d+)_year_(?P<year>\d{4})"
    r"(?:_(?P<repeated_county>\d+))?_"
    r"x(?P<x>-?\d+)_y(?P<y>-?\d+)_interval_(?P<interval>\d+)"
)


@dataclass(frozen=True)
class PatchSequence:
    county_id: str
    year: int
    x: int
    y: int
    intervals: tuple[int, ...]
    representative: Path

    @property
    def patch_id(self) -> str:
        return f"x{self.x}_y{self.y}"


@dataclass(frozen=True)
class PatchPoint:
    sequence: PatchSequence
    latitude: float
    longitude: float
    coordinate_fields: str


def discover_sequences(
    root: Path,
    expected_intervals: int = 7,
    include_incomplete: bool = False,
    years: set[int] | None = None,
    cohort_keys: set[tuple[str, int]] | None = None,
    interval_origin: str = "auto",
) -> tuple[list[PatchSequence], dict[str, Any]]:
    """Group NPZ files into county-year spatial patch sequences."""
    grouped: dict[tuple[str, int, int, int], dict[int, Path]] = defaultdict(dict)
    source_files = parsed_files = ignored_intervals = 0
    repeated_county_mismatch_files = 0
    year_filtered_files = cohort_filtered_files = 0
    interval_file_counts: dict[int, int] = defaultdict(int)
    filename_examples: list[str] = []
    unmatched_filename_examples: list[str] = []
    repeated_county_mismatch_examples: list[str] = []
    for path in root.rglob("*.npz"):
        source_files += 1
        if len(filename_examples) < 5:
            filename_examples.append(path.name)
        match = PATCH_RE.search(path.name)
        if match is None:
            if len(unmatched_filename_examples) < 5:
                unmatched_filename_examples.append(path.name)
            continue
        county = match.group("county").zfill(5)
        repeated_county = match.group("repeated_county")
        if repeated_county is not None and repeated_county.zfill(5) != county:
            repeated_county_mismatch_files += 1
            if len(repeated_county_mismatch_examples) < 5:
                repeated_county_mismatch_examples.append(path.name)
            continue
        parsed_files += 1
        year = int(match.group("year"))
        if years is not None and year not in years:
            year_filtered_files += 1
            continue
        if cohort_keys is not None and (county, year) not in cohort_keys:
            cohort_filtered_files += 1
            continue
        interval = int(match.group("interval"))
        interval_file_counts[interval] += 1
        key = (
            county,
            year,
            int(match.group("x")),
            int(match.group("y")),
        )
        existing = grouped[key].get(interval)
        if existing is None or str(path) < str(existing):
            grouped[key][interval] = path

    zero_based = set(range(expected_intervals))
    one_based = set(range(1, expected_intervals + 1))
    complete_zero = sum(zero_based.issubset(values) for values in grouped.values())
    complete_one = sum(one_based.issubset(values) for values in grouped.values())
    if interval_origin == "zero":
        required = zero_based
        selected_origin = "zero"
    elif interval_origin == "one":
        required = one_based
        selected_origin = "one"
    elif complete_one > complete_zero:
        required = one_based
        selected_origin = "one"
    else:
        required = zero_based
        selected_origin = "zero"
    ignored_intervals = sum(
        count for interval, count in interval_file_counts.items() if interval not in required
    )
    complete = sum(required.issubset(by_interval) for by_interval in grouped.values())
    sequences: list[PatchSequence] = []
    for (county, year, x, y), by_interval in sorted(grouped.items()):
        expected_paths = {
            interval: path for interval, path in by_interval.items() if interval in required
        }
        intervals = tuple(sorted(expected_paths))
        if not intervals:
            continue
        if not include_incomplete and not required.issubset(by_interval):
            continue
        representative_interval = min(intervals)
        sequences.append(
            PatchSequence(
                county_id=county,
                year=year,
                x=x,
                y=y,
                intervals=intervals,
                representative=expected_paths[representative_interval],
            )
        )

    return sequences, {
        "source_npz_files": source_files,
        "filename_pattern_matches": parsed_files,
        "repeated_county_mismatch_files": repeated_county_mismatch_files,
        "year_filtered_npz_files": year_filtered_files,
        "cohort_filtered_npz_files": cohort_filtered_files,
        "ignored_out_of_range_intervals": ignored_intervals,
        "discovered_patch_sequences": len(grouped),
        "complete_patch_sequences": complete,
        "selected_patch_sequences_before_sampling": len(sequences),
        "expected_intervals": expected_intervals,
        "interval_origin_requested": interval_origin,
        "interval_origin_selected": selected_origin,
        "complete_sequences_if_zero_based": complete_zero,
        "complete_sequences_if_one_based": complete_one,
        "interval_file_histogram": {
            str(interval): count for interval, count in sorted(interval_file_counts.items())
        },
        "npz_filename_examples": filename_examples,
        "unmatched_filename_examples": unmatched_filename_examples,
        "repeated_county_mismatch_examples": repeated_county_mismatch_examples,
    }


def load_cohort_keys(path: Path) -> set[tuple[str, int]]:
    """Read county-year keys from the benchmark split manifest or similar CSV."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = {str(name).strip().lower(): name for name in reader.fieldnames or []}
        fips_year = fields.get("fips_year") or fields.get("key")
        county = next(
            (fields[name] for name in ("county_id", "fips", "county_fips", "county")
             if name in fields),
            None,
        )
        year = fields.get("year")
        if fips_year is None and (county is None or year is None):
            raise ValueError(
                f"{path}: expected fips_year, or county_id/fips plus year columns"
            )
        keys: set[tuple[str, int]] = set()
        for row in reader:
            if fips_year is not None:
                match = re.fullmatch(r"\s*(\d{1,5})-(\d{4})\s*", row[fips_year])
                if match is None:
                    continue
                keys.add((match.group(1).zfill(5), int(match.group(2))))
            else:
                county_match = re.search(r"\d+", row[county])
                year_match = re.search(r"\d{4}", row[year])
                if county_match and year_match:
                    keys.add((county_match.group().zfill(5), int(year_match.group())))
    if not keys:
        raise ValueError(f"{path}: no valid county-year keys found")
    return keys


def read_npz_location(sequence: PatchSequence) -> PatchPoint:
    """Read latitude/longitude without unpickling object-valued metadata."""
    with np.load(sequence.representative, allow_pickle=False) as archive:
        if "raw_lat" in archive.files and "raw_lon" in archive.files:
            latitude = float(np.asarray(archive["raw_lat"]).reshape(()))
            longitude = float(np.asarray(archive["raw_lon"]).reshape(()))
            fields = "raw_lat/raw_lon"
        elif "coordinates" in archive.files:
            coordinates = np.asarray(archive["coordinates"], dtype=float).reshape(-1)
            if coordinates.size < 2:
                raise ValueError("coordinates must contain latitude and longitude")
            latitude, longitude = map(float, coordinates[:2])
            fields = "coordinates[latitude,longitude]"
        else:
            raise KeyError("NPZ contains neither raw_lat/raw_lon nor coordinates")

    if not (math.isfinite(latitude) and math.isfinite(longitude)):
        raise ValueError("non-finite latitude/longitude")
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ValueError(f"invalid latitude/longitude: {latitude}, {longitude}")
    return PatchPoint(sequence, latitude, longitude, fields)


def _safe_read(sequence: PatchSequence) -> tuple[PatchPoint | None, str | None]:
    try:
        return read_npz_location(sequence), None
    except Exception as exc:  # diagnostic context is written to the contract
        return None, f"{sequence.representative}: {type(exc).__name__}: {exc}"


def load_patch_points(
    sequences: list[PatchSequence], workers: int
) -> tuple[list[PatchPoint], list[str]]:
    """Read representative NPZ files in parallel while preserving order."""
    points: list[PatchPoint] = []
    errors: list[str] = []
    if workers == 1:
        results = map(_safe_read, sequences)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        results = executor.map(_safe_read, sequences)
    try:
        for point, error in results:
            if point is not None:
                points.append(point)
            if error is not None:
                errors.append(error)
    finally:
        if workers != 1:
            executor.shutdown(wait=True)
    return points, errors


def _feature_fips(properties: dict[str, Any]) -> str | None:
    for key in ("GEOID", "geoid", "FIPS", "fips", "county_id"):
        value = properties.get(key)
        if value is not None and re.fullmatch(r"\d{1,5}", str(value).strip()):
            return str(value).strip().zfill(5)
    state = properties.get("STATEFP") or properties.get("statefp")
    county = properties.get("COUNTYFP") or properties.get("countyfp")
    if state is not None and county is not None:
        return f"{str(state).zfill(2)}{str(county).zfill(3)}"
    return None


def _geometry_segments(geometry: dict[str, Any] | None) -> Iterable[np.ndarray]:
    """Yield longitude/latitude line segments from GeoJSON geometry."""
    if not geometry:
        return
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon":
        for ring in coordinates or []:
            array = np.asarray(ring, dtype=float)
            if array.ndim == 2 and array.shape[0] >= 2 and array.shape[1] >= 2:
                yield array[:, :2]
    elif geometry_type == "MultiPolygon":
        for polygon in coordinates or []:
            for ring in polygon:
                array = np.asarray(ring, dtype=float)
                if array.ndim == 2 and array.shape[0] >= 2 and array.shape[1] >= 2:
                    yield array[:, :2]
    elif geometry_type == "LineString":
        array = np.asarray(coordinates, dtype=float)
        if array.ndim == 2 and array.shape[0] >= 2 and array.shape[1] >= 2:
            yield array[:, :2]
    elif geometry_type == "MultiLineString":
        for line in coordinates or []:
            array = np.asarray(line, dtype=float)
            if array.ndim == 2 and array.shape[0] >= 2 and array.shape[1] >= 2:
                yield array[:, :2]
    elif geometry_type == "GeometryCollection":
        for child in geometry.get("geometries", []):
            yield from _geometry_segments(child)


def load_boundaries(
    path: Path, county_ids: set[str], all_boundaries: bool
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Read county outlines from a GeoJSON FeatureCollection."""
    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("type") != "FeatureCollection":
        raise ValueError(f"{path}: expected a GeoJSON FeatureCollection")

    features = document.get("features", [])
    selected: list[dict[str, Any]] = []
    for feature in features:
        fips = _feature_fips(feature.get("properties") or {})
        if all_boundaries or fips is None or fips in county_ids:
            selected.append(feature)
    segments = [
        segment
        for feature in selected
        for segment in _geometry_segments(feature.get("geometry"))
    ]
    # If a nonstandard property schema prevented FIPS matching, fall back to
    # the complete geometry rather than producing a blank map.
    if not segments and not all_boundaries:
        selected = features
        segments = [
            segment
            for feature in selected
            for segment in _geometry_segments(feature.get("geometry"))
        ]
    crs = document.get("crs")
    return segments, {
        "geojson_features": len(features),
        "plotted_geojson_features": len(selected),
        "geojson_crs": crs,
    }


def deterministic_subsample(
    sequences: list[PatchSequence], maximum: int | None
) -> list[PatchSequence]:
    if maximum is None or len(sequences) <= maximum:
        return sequences
    positions = np.linspace(0, len(sequences) - 1, maximum, dtype=int)
    return [sequences[int(position)] for position in positions]


def write_points_csv(path: Path, points: list[PatchPoint], npz_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "county_id", "year", "patch_id", "x", "y", "latitude",
                "longitude", "n_intervals", "intervals", "source_npz",
            ),
        )
        writer.writeheader()
        for point in points:
            sequence = point.sequence
            try:
                source = sequence.representative.relative_to(npz_root)
            except ValueError:
                source = sequence.representative
            writer.writerow(
                {
                    "county_id": sequence.county_id,
                    "year": sequence.year,
                    "patch_id": sequence.patch_id,
                    "x": sequence.x,
                    "y": sequence.y,
                    "latitude": f"{point.latitude:.8f}",
                    "longitude": f"{point.longitude:.8f}",
                    "n_intervals": len(sequence.intervals),
                    "intervals": ";".join(map(str, sequence.intervals)),
                    "source_npz": str(source),
                }
            )


def plot_map(
    points: list[PatchPoint],
    segments: list[np.ndarray],
    output: Path,
    marker_size: float,
    alpha: float,
    colour_by: str,
    width: float,
    height: float,
    dpi: int,
    title: str,
) -> None:
    try:
        import matplotlib
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting; install it with "
            "`python -m pip install matplotlib`"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, axis = plt.subplots(figsize=(width, height), dpi=dpi)
    if segments:
        axis.add_collection(
            LineCollection(
                segments,
                colors="#8a8a8a",
                linewidths=0.28,
                alpha=0.75,
                zorder=1,
            )
        )

    if colour_by == "year":
        palette = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")
        years = sorted({point.sequence.year for point in points})
        for index, year in enumerate(years):
            selected = [point for point in points if point.sequence.year == year]
            axis.scatter(
                [point.longitude for point in selected],
                [point.latitude for point in selected],
                s=marker_size,
                alpha=alpha,
                color=palette[index % len(palette)],
                edgecolors="none",
                label=str(year),
                rasterized=True,
                zorder=2,
            )
        axis.legend(
            title="County-year",
            frameon=False,
            loc="lower left",
            markerscale=2.0,
            ncols=min(4, len(years)),
        )
    else:
        axis.scatter(
            [point.longitude for point in points],
            [point.latitude for point in points],
            s=marker_size,
            alpha=alpha,
            color="#0072B2",
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )

    longitudes = np.asarray([point.longitude for point in points])
    latitudes = np.asarray([point.latitude for point in points])
    longitude_span = float(np.ptp(longitudes)) or 1.0
    latitude_span = float(np.ptp(latitudes)) or 1.0
    axis.set_xlim(float(longitudes.min() - 0.025 * longitude_span),
                  float(longitudes.max() + 0.025 * longitude_span))
    axis.set_ylim(float(latitudes.min() - 0.035 * latitude_span),
                  float(latitudes.max() + 0.035 * latitude_span))
    mean_latitude = float(latitudes.mean())
    axis.set_aspect(1.0 / math.cos(math.radians(mean_latitude)))
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.set_title(title)
    axis.text(
        0.995,
        0.015,
        f"{len(points):,} patch sequences",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#333333",
    )
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def no_match_message(counts: dict[str, Any], include_incomplete: bool) -> str:
    """Explain which discovery/filter stage removed all patch sequences."""
    lines = [
        "no patch sequences matched the requested filters",
        "scan summary:",
        f"  NPZ files found: {counts['source_npz_files']:,}",
        f"  filenames matching the expected pattern: "
        f"{counts['filename_pattern_matches']:,}",
        f"  files with conflicting repeated county FIPS: "
        f"{counts['repeated_county_mismatch_files']:,}",
        f"  files removed by --years: {counts['year_filtered_npz_files']:,}",
        f"  files removed by --cohort: {counts['cohort_filtered_npz_files']:,}",
        f"  patch groups after year/cohort filtering: "
        f"{counts['discovered_patch_sequences']:,}",
        f"  interval numbering selected: {counts['interval_origin_selected']}",
        f"  complete groups if 0--{counts['expected_intervals'] - 1}: "
        f"{counts['complete_sequences_if_zero_based']:,}",
        f"  complete groups if 1-based: {counts['complete_sequences_if_one_based']:,}",
        f"  observed interval histogram: {counts['interval_file_histogram']}",
    ]
    if counts["source_npz_files"] == 0:
        lines.append("The selected directory contains no .npz files.")
    elif counts["filename_pattern_matches"] == 0:
        lines.append(
            "The filenames do not match either supported form: "
            "county_<FIPS>_year_<YYYY>_x<X>_y<Y>_interval_<N>, or "
            "county_<FIPS>_year_<YYYY>_<FIPS>_x<X>_y<Y>_interval_<N>. "
            "Examples: " + ", ".join(counts["npz_filename_examples"])
        )
    elif counts["discovered_patch_sequences"] == 0:
        lines.append(
            "The year/cohort intersection is empty. Re-run without --cohort and "
            "--years to inspect the archive before applying the benchmark manifest."
        )
    elif not include_incomplete:
        lines.append(
            "No group contains the complete expected interval set. Auto-detection "
            "checked both 0-based and 1-based numbering. Use --include-incomplete "
            "only if plotting incomplete sequences is intentional."
        )
    if counts["unmatched_filename_examples"]:
        lines.append(
            "Unmatched filename examples: "
            + ", ".join(counts["unmatched_filename_examples"])
        )
    if counts["repeated_county_mismatch_examples"]:
        lines.append(
            "Conflicting repeated-county examples: "
            + ", ".join(counts["repeated_county_mismatch_examples"])
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--npz-dir", required=True, type=Path)
    parser.add_argument("--geometry", required=True, type=Path,
                        help="county-boundary GeoJSON in longitude/latitude coordinates")
    parser.add_argument(
        "--cohort",
        type=Path,
        help="optional CSV manifest used to retain the benchmark county-years",
    )
    parser.add_argument("--out", required=True, type=Path,
                        help="output figure (.png, .pdf, or .svg)")
    parser.add_argument("--years", nargs="+", type=int,
                        help="optional subset of county-years to plot")
    parser.add_argument("--expected-intervals", type=int, default=7)
    parser.add_argument(
        "--interval-origin",
        choices=("auto", "zero", "one"),
        default="auto",
        help="interval numbering convention; auto compares 0--6 with 1--7",
    )
    parser.add_argument("--include-incomplete", action="store_true",
                        help="also plot patch sequences missing one or more intervals")
    parser.add_argument("--all-boundaries", action="store_true",
                        help="draw every GeoJSON feature, not only sampled counties")
    parser.add_argument("--max-points", type=int,
                        help="deterministic point limit for previews")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--marker-size", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.28)
    parser.add_argument("--colour-by", choices=("none", "year"), default="none")
    parser.add_argument("--width", type=float, default=10.0)
    parser.add_argument("--height", type=float, default=7.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--title",
        default="Sentinel-2 patch-sequence locations in the U.S. Corn Belt",
    )
    args = parser.parse_args()

    if not args.npz_dir.is_dir():
        parser.error(f"NPZ directory does not exist: {args.npz_dir}")
    if not args.geometry.is_file():
        parser.error(f"GeoJSON does not exist: {args.geometry}")
    if args.cohort is not None and not args.cohort.is_file():
        parser.error(f"cohort manifest does not exist: {args.cohort}")
    if args.expected_intervals < 1:
        parser.error("--expected-intervals must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.max_points is not None and args.max_points < 1:
        parser.error("--max-points must be positive")
    if not 0 < args.alpha <= 1:
        parser.error("--alpha must be in (0, 1]")

    years = set(args.years) if args.years else None
    cohort_keys = load_cohort_keys(args.cohort) if args.cohort else None
    sequences, counts = discover_sequences(
        args.npz_dir,
        expected_intervals=args.expected_intervals,
        include_incomplete=args.include_incomplete,
        years=years,
        cohort_keys=cohort_keys,
        interval_origin=args.interval_origin,
    )
    sequences = deterministic_subsample(sequences, args.max_points)
    if not sequences:
        raise SystemExit(no_match_message(counts, args.include_incomplete))

    print(
        f"Reading coordinates from {len(sequences):,} representative NPZ files "
        f"with {args.workers} worker(s)...",
        flush=True,
    )
    points, errors = load_patch_points(sequences, args.workers)
    if not points:
        detail = errors[0] if errors else "unknown coordinate-reading error"
        raise SystemExit(f"no valid patch locations were read; first error: {detail}")

    county_ids = {point.sequence.county_id for point in points}
    segments, geometry_contract = load_boundaries(
        args.geometry, county_ids, args.all_boundaries
    )
    plot_map(
        points,
        segments,
        args.out,
        marker_size=args.marker_size,
        alpha=args.alpha,
        colour_by=args.colour_by,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        title=args.title,
    )

    points_csv = Path(str(args.out) + ".points.csv")
    contract_path = Path(str(args.out) + ".contract.json")
    write_points_csv(points_csv, points, args.npz_dir)
    coordinate_fields = sorted({point.coordinate_fields for point in points})
    contract = {
        **counts,
        **geometry_contract,
        "npz_directory": str(args.npz_dir.resolve()),
        "geometry": str(args.geometry.resolve()),
        "cohort_manifest": str(args.cohort.resolve()) if args.cohort else None,
        "cohort_manifest_county_years": len(cohort_keys) if cohort_keys else None,
        "figure": str(args.out.resolve()),
        "points_csv": str(points_csv.resolve()),
        "complete_only": not args.include_incomplete,
        "expected_intervals": args.expected_intervals,
        "year_filter": sorted(years) if years else None,
        "max_points": args.max_points,
        "plotted_patch_sequences": len(points),
        "plotted_counties": len(county_ids),
        "plotted_years": sorted({point.sequence.year for point in points}),
        "coordinate_fields": coordinate_fields,
        "coordinate_read_errors": len(errors),
        "coordinate_read_error_examples": errors[:10],
        "point_definition": "one point per county-year x/y patch sequence",
    }
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"wrote {points_csv}")
    print(f"wrote {contract_path}")
    print(
        f"plotted {len(points):,} patch sequences from {len(county_ids):,} counties; "
        f"coordinate read errors: {len(errors):,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
