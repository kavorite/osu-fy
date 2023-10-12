import itertools as it
import multiprocessing.pool as mpp
import re
from glob import glob
from typing import NamedTuple

import audio_decode
import fire
import jax
import jax.tree_util as jtu
import numpy as np
import polars as pl
import polars.selectors as pls
import scipy.stats as sst


class Difficulty(NamedTuple):
    "Difficulty ratings. All values lie on [0, 10]."
    circle_size: jax.Array
    drain_rate: jax.Array
    overall_difficulty: jax.Array
    slider_multiplier: jax.Array
    slider_tick_rate: jax.Array
    approach_rate: jax.Array


class Beatmap(NamedTuple):
    positions: jax.Array
    is_new_combo: jax.Array
    is_new_curve: jax.Array
    is_new_timing: jax.Array
    num_repeats: jax.Array
    hit_types: jax.Array
    slider_types: jax.Array
    difficulties: Difficulty


def standardize(x: pl.Expr, eps=1e-5):
    shift = x.mean()
    scale = x.std().clip_min(eps)
    return (x - shift) / scale


def snake_case(s):
    return re.sub(r"[A-Z]", r"_\g<0>", s).lower().lstrip("_")


# def bms_audio(path):
#     audio, hits = audio_decode.extract(path)
#     bms_id = int(re.search(r"([0-9]+).zip$", path)[1])
#     return pl.DataFrame({"beatmap_set_id": bms_id, "sample": [audio]}), hits


def bms_hits(meta: pl.LazyFrame, hits: pl.LazyFrame, bms_id):
    return (
        hits.filter(pl.col("beatmap_set_id") == pl.lit(bms_id))
        .join(meta.select("beatmap_id", "mode", "difficulty"), on="beatmap_id")
        .filter(pl.col("mode") == pl.lit("osu"))
        .sort("beatmap_id", "start_time")
        .collect()
    )


def scan_meta(path):
    return pl.scan_ndjson(path, infer_schema_length=1).rename(
        {"id": "beatmap_set_id", "favourite_count": "fave_count"}
    )


def scan_maps(path):
    meta = scan_meta(path)
    maps = (
        meta.select("beatmaps")
        .explode("beatmaps")
        .unnest("beatmaps")
        .rename({"beatmapset_id": "beatmap_set_id", "id": "beatmap_id"})
        .select(pl.exclude(meta.columns), "beatmap_set_id")
    )
    return maps.join(
        meta.drop("beatmaps"), on="beatmap_set_id", how="left"
    ).with_columns(
        (pl.col("fave_count") / pl.col("playcount")).clip(0, 1).alias("fave_rate"),
    )


def scan_hits(path):
    curve_types = [None] + list("PLBCS")
    curve_types = pl.DataFrame(
        {
            "curve_type": curve_types,
            "type_code": np.arange(len(curve_types), dtype=np.int8),
        },
    )
    hits = (
        pl.scan_parquet(path, parallel="row_groups")
        .explode("path")
        .unnest("path")
        .with_columns(pls.float().cast(pl.Float32))
        .select(pl.all().map_alias(snake_case))
        .select(
            pl.exclude("position", "type"),
            pl.when(pl.col("position").is_not_null())
            .then(pl.col("position"))
            .otherwise(pl.col("start_position"))
            .alias("position"),
        )
        .select(pl.all().map_alias(snake_case))
        .with_columns(
            pl.when(pl.col("hit_type") & (1 << 3) == 0)
            .then(pl.col("curve_type"))
            .otherwise(pl.lit("S")),
        )
        .join(curve_types.lazy(), on="curve_type")
    )
    return hits


def batch_array_trees(batch_size, trees_iter):
    buffer = None
    treedef = None

    def _buf_len():
        if buffer is not None:
            return buffer[0].shape[0]
        else:
            return 0

    def _is_array(a):
        return isinstance(a, np.ndarray)

    def _concat(*args):
        return np.concatenate(args)

    def _chunk_split(a):
        return a[:batch_size], a[batch_size:]

    def _pad_right(a):
        pad_rem = batch_size % a.shape[0]
        if pad_rem != 0:
            pad_len = batch_size - pad_rem
            padding = [(0, 0)] * a.ndim
            padding[0] = (0, pad_len)
            return np.pad(a, padding)
        else:
            return a

    for tree in trees_iter:
        leaves, _def = jtu.tree_flatten(tree, is_leaf=_is_array)
        if treedef is not None and _def != treedef:
            raise ValueError(f"initial treedef {treedef} supplanted by {_def}")
        else:
            treedef = _def
        if buffer is None:
            buffer = leaves
        buffer = tuple(map(_concat, buffer, leaves))
        while _buf_len() >= batch_size:
            output, buffer = zip(*map(_chunk_split, buffer))
            output = jtu.tree_unflatten(treedef, output)
            yield output

    if _buf_len() > 0:
        buffer = tuple(map(_pad_right, buffer))
        yield jtu.tree_unflatten(treedef, buffer)


class Ratings(NamedTuple):
    challenge: jax.Array
    fave_rate: jax.Array


class Batch(NamedTuple):
    samples: jax.Array
    seq_ids: jax.Array
    ratings: Ratings
    beatmap: Beatmap


def bms_chunks(
    osz_paths,
    lazy_maps: pl.LazyFrame,
    mapper_fn=map,
    diff_dist_cls=sst.expon,
    fave_dist_cls=sst.expon,
):
    maps = lazy_maps.with_columns(
        (pl.col("fave_count") / pl.col("play_count")).alias("fave_rate")
    )
    fave_dist = fave_dist_cls(
        *fave_dist_cls.fit(maps.select("fave_rate").collect().to_series())
    )
    diff_dist = diff_dist_cls(
        *diff_dist_cls.fit(maps.select("difficulty_rating").collect().to_series())
    )

    def _load(path):
        try:
            audio, hits = audio_decode.extract(path)
        except:
            return (None, None)
        slider_types = pl.DataFrame(
            {
                "slider_type": ["bezier", "catmull", "circle", "linear"],
                "slider_type_code": np.arange(4, dtype=np.int8),
            },
        )
        hit_types = pl.DataFrame(
            {
                "hit_type": ["slider", "spinner", "held", "hit"],
                "hit_type_code": np.arange(4, dtype=np.int8),
            }
        )

        hits = (
            pl.read_ndjson(hits.encode("utf8"))
            .lazy()
            .with_columns((pl.col("start_time") + pl.col("ctl_index")).alias("offset"))
            .join(slider_types.lazy(), on="slider_type")
            .join(hit_types.lazy(), on="hit_type")
            .join(
                maps.select("beatmap_id", "mode", "fave_rate", "difficulty_rating"),
                on="beatmap_id",
            )
            .filter(pl.col("mode") == "osu")
            .with_columns(
                (pl.col("timing") != pl.col("timing").shift_and_fill(1)).alias(
                    "is_new_timing"
                ),
            )
            .sort("beatmap_id", "start_time")
            .collect()
        )
        return audio, hits

    screen_size = np.array([640, 480], dtype=np.float32)

    for seq_id, (all_samples, set_hits) in enumerate(mapper_fn(_load, osz_paths)):
        if all_samples is None and set_hits is None:
            continue
        elif len(set_hits) == 0:
            continue

        def _positions(chunk_hits):
            chunk_poses = (
                chunk_hits.select("position")
                .unnest("position")
                .select(pl.all().cast(pl.Float32))
                .to_numpy()
            )
            chunk_poses -= screen_size / 2
            chunk_poses /= screen_size
            return chunk_poses

        def _difficulties(chunk_hits):
            chunk_difficulty = (
                chunk_hits.lazy()
                .select("difficulty")
                .unnest("difficulty")
                .select((pl.all().cast(pl.Float32) - 5.0) / 10.0)
                .collect()
                .to_numpy()
            )
            return chunk_difficulty

        pad_len = 48 - len(all_samples) % 48
        if pad_len != 0:
            all_samples = np.pad(all_samples, pad_width=[(0, pad_len)])
        for bmid, hits in set_hits.group_by("beatmap_id"):
            seq_id = (bmid + seq_id) % np.iinfo(np.int8).max
            samples = all_samples
            chunk = hits
            repeats = chunk.select(
                pl.col("offset").diff().fill_null(pl.col("offset").first())
            ).to_series()
            repeats = chunk.select(
                pl.col("offset").shift(-1) - pl.col("offset")
            ).to_series()
            repeats[-1] = len(samples) // 48 - repeats[:-1].sum()
            indices = chunk.select("offset").to_series()
            difficulties = Difficulty(
                *np.repeat(_difficulties(chunk), repeats, axis=-2).T
            )
            positions = np.zeros([len(samples) // 48, 2], dtype=np.float32)
            positions[indices] = _positions(chunk)
            is_new_combo = np.zeros([len(samples) // 48], dtype=bool)
            is_new_combo[indices] = chunk["new_combo"]
            is_new_curve = np.zeros([len(samples) // 48], dtype=bool)
            is_new_curve[indices] = chunk.select(pl.col("ctl_index") == 0).to_series()
            num_repeats = np.zeros([len(samples) // 48], dtype=np.int8)
            num_repeats[indices] = repeats
            hit_types = np.zeros([len(samples) // 48], dtype=np.int8)
            hit_types[indices] = chunk.select("hit_type_code").to_series()
            slider_types = np.zeros([len(samples) // 48], dtype=np.int8)
            slider_types[indices] = chunk.select("slider_type_code").to_series()
            is_new_timing = np.zeros([len(samples) // 48], dtype=bool)
            is_new_timing[indices] = chunk.select("is_new_timing").to_series()
            outputs = Beatmap(
                positions=positions,
                is_new_combo=is_new_combo,
                is_new_curve=is_new_curve,
                is_new_timing=is_new_timing,
                num_repeats=num_repeats,
                difficulties=difficulties,
                slider_types=slider_types,
                hit_types=hit_types,
            )
            seq_ids = np.full(
                shape=[len(samples) // 48], fill_value=seq_id, dtype=np.int8
            )
            fave_rate, difficulty_rating = hits.select(
                pl.col("fave_rate", "difficulty_rating").first()
            )
            ratings = np.zeros(
                [2, len(samples) // 48],
                dtype=np.float32,
            )
            samples = samples.reshape(-1, 48)
            ratings[0] = fave_dist.cdf(fave_rate) - 0.5
            ratings[1] = diff_dist.cdf(difficulty_rating) - 0.5
            ratings = Ratings(*ratings)
            chunk = Batch(
                samples=samples,
                seq_ids=seq_ids,
                ratings=ratings,
                beatmap=outputs,
            )
            yield chunk


if __name__ == "__main__":
    import concurrent.futures as cft
    import time

    import rich.progress as rp

    osz_paths = glob("./osu-dl/beatmapsets/*.zip")[:100]
    lazy_maps = scan_maps("./osu-dl/meta.jsonl")
    lazy_hits = scan_hits("./osu-dl/hits.parquet")
    # for x in batch_array_trees(
    #     32 * 1000 * 256, bms_chunks(osz_paths, lazy_maps, mapper_fn=pool.map)
    # ):
    start_time = time.time()
    total_ms = 0
    elapsed = 0

    with cft.ThreadPoolExecutor(32) as pool:
        for i, x in enumerate(
            batch_array_trees(
                12288 * 256,
                rp.track(
                    bms_chunks(osz_paths, lazy_maps, mapper_fn=pool.map),
                    total=len(osz_paths),
                ),
            )
        ):
            total_ms += x[0].shape[0]
            elapsed = time.time() - start_time
            pass
    pass
    # for x in cycle_hit_aligned_audio(
    #     cache_root="./audio-cache",
    #     meta_path="./osu-dl/meta.jsonl",
    #     hits_path="./osu-dl/hits.parquet",
    #     batch_size=8,
    #     seq_length=48000 * 32,
    #     seed=42,
    # ):
    #     pass
    #     pass
