import multiprocessing.pool as mpp
import re
from glob import glob
from zipfile import ZipFile

import av
import fire
import jax.numpy as jnp
import numpy as np
import polars as pl
import rich.progress as rp
import scipy.stats as sst
from flax.training import train_state
from miniaudio import decode

from model import Beatmap, Difficulty, Mapper


def beatmapset_audio(archive_path):
    with ZipFile(archive_path) as archive:
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


def standardize(x: pl.Expr, eps=1e-5):
    shift = x.mean()
    scale = x.std().clip_min(eps)
    return (x - shift) / scale


def snake_case(s):
    return re.sub(r"[A-Z]", r"_\g<0>", s).lower().lstrip("_")


def main(
    width=128,
    depth=6,
    dtype="bfloat16",
    meta_path="osu-dl/meta.jsonl",
    oszs_path="osu-dl/beatmapsets",
    cache_root="./audio-cache",
    seq_length=32 * 48000,
    batch_size=8,
):
    from istrm import cycle_streams

    for ledger, chunk in cycle_streams(cache_root, seq_length, batch_size):
        pass

    # start = time.time()
    # _audio_by_id(beatmapset_paths[0])
    # end = time.time()

    class TrainState(train_state.TrainState):
        loss: float

    # hits = hits.join(curve_types, on="curve_type")
    mapper = Mapper(width, depth, dtype=getattr(jnp, dtype))
    joint = maps.join(hits, left_on="id", right_on="beatmap_id")
    # difficulty components
    pass


if __name__ == "__main__":
    fire.Fire(main)
