import multiprocessing.pool as mpp
import re
from glob import glob
from zipfile import ZipFile

import fire
import numpy as np
import polars as pl
import rich.progress as rp
import scipy.stats as sst
from miniaudio import decode


def standardize(x: pl.Expr, eps=1e-5):
    shift = x.mean()
    scale = x.std().clip_min(eps)
    return (x - shift) / scale


def snake_case(s):
    return re.sub(r"[A-Z]", r"_\g<0>", s).lower().lstrip("_")


def bms_audio(bms_path):
    with ZipFile(bms_path) as archive:
        audio_files = (
            entry for entry in archive.filelist if entry.filename.endswith(".mp3")
        )
        audio_entry = next(audio_files, None)
        if audio_entry is None:
            return None
        istrm = archive.open(audio_entry).read()
        try:
            audio = decode(istrm, nchannels=1, sample_rate=48000)
        except:
            return None
        audio = np.frombuffer(audio.samples, dtype=np.int16)
        return audio


def bms_audio_frame(path):
    samples = bms_audio(path)
    if samples is None:
        return None
    else:
        bms_id = int(re.search(r"([0-9]+).zip$", path)[1])
        return pl.DataFrame({"bms_ids": bms_id, "samples": [samples]})


def cycle_streams(cache_root, batch_size, seq_length):
    chunk_size = batch_size * seq_length
    entries = (
        pl.scan_parquet(f"{cache_root}/*.parquet", parallel="row_groups", rechunk=False)
        .with_row_count("seq_ids")
        .with_columns((pl.col("seq_ids") % 256).cast(pl.UInt8))
    )
    ledger = entries.select(
        "bms_ids",
        "seq_ids",
        pl.col("samples").list.lengths().cast(pl.UInt64).alias("lengths"),
    ).collect()
    stride = 0
    lengths = ledger["lengths"]
    total_length = lengths.sum()
    while True:
        offsets = ((lengths.cumsum() - lengths + stride) % total_length).alias(
            "offsets"
        )
        chunk_ids = (offsets + stride) // chunk_size
        for chunk_id in range(offsets[-1] // chunk_size):
            start = np.searchsorted(chunk_ids, chunk_id, side="left")
            end = np.searchsorted(chunk_ids, chunk_id, side="right")
            chunk_stride = lengths[start:end].sum()
            chunk_length = min(chunk_stride, chunk_size)
            if chunk_length < chunk_size:
                continue
            else:
                chunk_ledger = ledger.slice(start, end - start).with_columns(
                    offsets[start:end]
                )
                chunk = (
                    entries.slice(start, end - start)
                    .select("samples", "seq_ids")
                    .explode("samples")
                    .head(chunk_size)
                    .collect()
                )
                stride = (offsets[-1] + chunk_stride) % chunk_size
                yield chunk_ledger, chunk


def cycle_hit_aligned_audio(
    cache_root, meta_path, hits_path, oszs_path, batch_size, seq_length, seed
):
    prng = np.random.default_rng(seed)
    sets = (
        pl.scan_ndjson(meta_path, infer_schema_length=1)
        .select("id", "favourite_count", "beatmaps")
        .collect()
    )
    maps = (
        sets.select("beatmaps")
        .explode("beatmaps")
        .unnest("beatmaps")
        .filter((pl.col("playcount") > 0))
        .join(sets.drop("beatmaps"), left_on="beatmapset_id", right_on="id")
        .with_columns(
            (
                standardize(pl.col("difficulty_rating")),
                (pl.col("favourite_count") / pl.col("playcount"))
                .clip(0, 1)
                .alias("fav_rate"),
                (pl.col("passcount") / pl.col("playcount")).alias("win_rate"),
            )
        )
    )

    favs = sst.expon(*sst.expon.fit(maps["fav_rate"]))
    maps = maps.with_columns(
        pl.Series("fav_rate", sst.norm.ppf(favs.cdf(maps["fav_rate"])))
    )

    hits = (
        pl.scan_parquet("osu-dl/hits.parquet")
        .explode("path")
        .unnest("path")
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
            .otherwise(pl.lit("S"))
        )
        .unnest("difficulty")
        .collect()
    )
    # spinners
    curve_types = [None] + list("PLBCS")
    curve_types = pl.DataFrame(
        {"curve_type": curve_types, "type_code": range(len(curve_types))}
    )

    for ledger, samples in cycle_streams(cache_root, batch_size, seq_length):
        final_offset = ledger["lengths"][-1] - (len(samples) - ledger["offsets"][-1])
        final_offset_ms = final_offset // 48
        final_bms_id = ledger["bms_ids"][-1]
        final_offset_mask = (pl.col("beatmap_set_id") != final_bms_id) | (
            pl.col("start_time") > final_offset_ms
        )
        chunk_beatmaps = hits.filter(
            pl.col("beatmap_set_id").is_in(ledger["bms_ids"])
        ).select("beatmap_id")
        chunk_beatmaps = chunk_beatmaps.sample(
            n=len(chunk_beatmaps), shuffle=True, seed=prng.bit_generator.random_raw()
        ).to_series()
        map_chunk = maps.filter(
            pl.int_range(0, pl.count())
            .shuffle(seed=prng.bit_generator.random_raw())
            .over("beatmapset_id")
            == 0
        )
        hit_chunk = hits.filter(
            pl.col("beatmap_id").is_in(map_chunk["id"]) & final_offset_mask
        )
        yield hit_chunk, samples
        pass


def build_cache(
    oszs_path="./osu-dl/beatmapsets",
    cache_root="./audio-cache",
):
    osz_paths = glob(f"{oszs_path}/*.zip")

    def _write_audio(osz_path):
        frame = bms_audio_frame(osz_path)
        if frame is not None:
            bms_id = frame.select(pl.col("bms_ids").first()).item()
            frame.write_parquet(f"{cache_root}/{bms_id}.parquet")

    with mpp.ThreadPool() as pool:
        for _ in rp.track(
            pool.imap_unordered(_write_audio, osz_paths),
            total=len(osz_paths),
        ):
            pass


if __name__ == "__main__":
    # fire.Fire(build_cache)
    for x in cycle_hit_aligned_audio(
        cache_root="./audio-cache",
        oszs_path="./osu-dl/beatmapsets/",
        meta_path="./osu-dl/meta.jsonl",
        hits_path="./osu-dl/hits.parquet",
        batch_size=256,
        seq_length=48000 * 32,
        seed=42,
    ):
        pass
