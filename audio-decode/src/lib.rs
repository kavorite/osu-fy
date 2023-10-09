use std::fmt::Write;

use json_writer::JSONObjectWriter;
use numpy::{IntoPyArray, PyArray1};
use osuparse::{
    DifficultySection, HitCircle, HitObject, HoldNote, MetadataSection, Slider, SliderType,
    Spinner, TimingPoint,
};
use pyo3::{
    exceptions::{PyNotImplementedError, PyValueError},
    prelude::*,
};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

use symphonia::core::{
    audio::{AudioBuffer, AudioBufferRef, Channels},
    codecs::{Decoder, DecoderOptions, CODEC_TYPE_NULL},
    errors::Error as SymphoniaError,
    formats::{FormatReader, Packet},
    io::MediaSourceStream,
    meta::{Limit, MetadataOptions},
    probe::Hint,
    sample::Sample,
};
use symphonia::default::formats::{MpaReader, OggReader};

use error_chain::error_chain;

error_chain! {
    errors {
        NoBeatmapFilesFound {
            display("no .osu files present in archive")
        }
        MissingAudio(file_name: String) {
            display("audio file '{file_name}' not present in archive")
        }
        UnknownFrames {
            display("length of audio file not found in container")
        }
        UnknownRate {
            display("sample-rate of audio file not found in container")
        }
        UnsupportedCodec {
            display("no supported audio tracks present in container")
        }
        UnsupportedSampleFormat(fmt: &'static str) {
            description("unsupported sample format")
            display("unsupported sample format '{fmt:?}'")
        }
    }
    foreign_links {
        Unarchive(zip::result::ZipError);
        Decode(SymphoniaError);
        Resample(rubato::ResampleError);
        Io(std::io::Error);
        Py(PyErr);
        Utf8(std::str::Utf8Error);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_count() {
        let data = include_bytes!("../audio.ogg");
        let cursor = std::io::Cursor::new(data);
        let boxed = Box::new(cursor);
        let istrm = MediaSourceStream::new(boxed, Default::default());
        let res = _decode("ogg", istrm);
        println!("{:#?}", res);
        assert!(res.is_ok());
    }
}

impl Into<PyErr> for Error {
    fn into(self) -> PyErr {
        match self.0 {
            ErrorKind::UnsupportedCodec => PyNotImplementedError::new_err(self.to_string()),
            ErrorKind::UnsupportedSampleFormat(_) => {
                PyNotImplementedError::new_err(self.to_string())
            }
            ErrorKind::Decode(err) => PyValueError::new_err(format!("decode: {err}")),
            ErrorKind::Resample(err) => PyValueError::new_err(format!("resample: {err}")),
            ErrorKind::Py(err) => err,
            _ => PyValueError::new_err(self.to_string()),
        }
    }
}
macro_rules! match_enum_right {
    (
        $enum_name:ident {
            $(
                $variant:ident,
            )*
        }
    ) => {
        match $enum_name {
            $(
                $enum_name::$variant => {
                    stringify!($variant).contains("RIGHT")
                },
            )*
        }
    }
}

fn phase_shift(channel: Channels) -> f32 {
    match channel {
        Channels::FRONT_RIGHT
        | Channels::FRONT_RIGHT_CENTRE
        | Channels::FRONT_RIGHT_HIGH
        | Channels::FRONT_RIGHT_WIDE
        | Channels::REAR_RIGHT
        | Channels::REAR_RIGHT_CENTRE
        | Channels::SIDE_RIGHT
        | Channels::TOP_FRONT_RIGHT
        | Channels::TOP_REAR_RIGHT => -1f32,
        _ => 1f32,
    }
}

fn load_mono_buf<S>(src: impl AsRef<AudioBuffer<S>>, dst: &mut [f32])
where
    S: Sample + num::Zero,
    f64: From<S>,
{
    let planes = src.as_ref().planes();
    let channels = src.as_ref().spec().channels;
    let scale = if S::zero() == S::MID {
        1 << (S::EFF_BITS - 1)
    } else {
        1 << S::EFF_BITS
    } as f64;
    let shift: f64 = S::MID.into();
    for (plane, channel) in planes.planes().iter().zip(channels.iter()) {
        for (i, x) in plane.iter().copied().enumerate() {
            let value: f64 = x.into();
            dst[i] += phase_shift(channel) * ((value - shift) / scale) as f32;
        }
    }
}

fn load_mono(src: &AudioBufferRef, dst: &mut [f32]) {
    use AudioBufferRef::*;
    match src {
        F32(src) => load_mono_buf(src, dst),
        F64(src) => load_mono_buf(src, dst),
        S8(src) => load_mono_buf(src, dst),
        S16(src) => load_mono_buf(src, dst),
        S24(src) => {
            let channels = src.as_ref().spec().channels;
            let planes = src.as_ref().planes();
            for (plane, channel) in planes.planes().iter().zip(channels.iter()) {
                for (i, value) in plane.iter().copied().enumerate() {
                    let value: f64 = value.0.into();
                    let scale = (1 << 23) as f64;
                    dst[i] += phase_shift(channel) * (value / scale) as f32;
                }
            }
        }
        S32(src) => load_mono_buf(src, dst),
        U8(src) => load_mono_buf(src, dst),
        U16(src) => load_mono_buf(src, dst),
        U24(src) => {
            let channels = src.as_ref().spec().channels;
            let planes = src.as_ref().planes();
            let shift: f64 = (1 << 23) as f64;
            let scale = (1 << 24) as f64;
            for (plane, channel) in planes.planes().iter().zip(channels.iter()) {
                for (i, value) in plane.iter().copied().enumerate() {
                    let value: f64 = value.0.into();
                    dst[i] += phase_shift(channel) * ((value - shift) / scale) as f32;
                }
            }
        }
        U32(src) => load_mono_buf(src, dst),
    }
}

struct MonoStream {
    packets: Box<dyn FormatReader>,
    resampler: Option<SincFixedIn<f32>>,
    track_id: u32,
    decoder: Box<dyn Decoder>,
    buffer: Vec<f32>,
    output: Vec<f32>,
}

macro_rules! try_harder {
    ($x:expr) => {
        match $x {
            Some(Ok(value)) => value,
            Some(Err(err)) => return Some(Err(err.into())),
            None => return None,
        }
    };
}

impl MonoStream {
    fn new(packets: Box<dyn FormatReader>, track_id: u32) -> Self {
        let track = packets
            .tracks()
            .iter()
            .filter(|track| track.id == track_id)
            .next()
            .expect("track id not found");
        let resampler = None;
        let decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions { verify: false })
            .unwrap();
        let buffer = Vec::with_capacity(1152);
        let output = Vec::new();
        MonoStream {
            packets,
            track_id,
            resampler,
            decoder,
            buffer,
            output,
        }
    }

    fn resampler(resample_ratio: f64) -> SincFixedIn<f32> {
        let chunk_size: usize = 576;
        let max_resample_ratio_relative: f64 = 2.0;
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        SincFixedIn::<f32>::new(
            resample_ratio,
            max_resample_ratio_relative,
            params,
            chunk_size,
            1,
        )
        .unwrap()
    }

    fn set_resample_ratio(&mut self, resample_ratio: f64) -> Result<()> {
        if let Some(ref mut resampler) = self.resampler {
            resampler
                .set_resample_ratio(resample_ratio, false)
                .map_err(Error::from)
        } else {
            self.resampler = if resample_ratio != 1.0 {
                self.output
                    .reserve((self.output.capacity() as f64 * resample_ratio).ceil() as usize);
                Some(Self::resampler(resample_ratio))
            } else {
                None
            };
            Ok(())
        }
    }

    fn with_resample_ratio(mut self, resample_ratio: f64) -> Self {
        self.set_resample_ratio(resample_ratio).unwrap();
        self
    }

    fn next_packet(&mut self) -> Option<Result<Packet>> {
        match self.packets.next_packet() {
            Ok(packet) => Some(Ok(packet)),
            Err(SymphoniaError::IoError(err))
                if err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                None
            }
            // Err(SymphoniaError::DecodeError(_) | SymphoniaError::IoError(_)) => continue,
            Err(err) => Some(Err(err.into())),
        }
    }

    fn consume_next<'a>(&'a mut self) -> Option<Result<std::vec::Drain<'a, f32>>> {
        let packet = loop {
            let packet = try_harder!(self.next_packet());
            if packet.track_id() == self.track_id {
                break packet;
            }
        };
        {
            let packet = try_harder!(self.decoder.decode(&packet).map(Some).transpose());
            let packet_spec = packet.spec();
            let frames = packet.frames();
            self.buffer.resize(frames + self.buffer.len(), 0f32);
            load_mono(&packet, &mut self.buffer);
            let ratio = 48000.0 / packet_spec.rate as f64;
            try_harder!(self.set_resample_ratio(ratio).map(Some).transpose());
        }

        let buffer = if let Some(ref mut resampler) = self.resampler {
            let output = &mut self.output;
            let buffer = &mut self.buffer;
            buffer.resize(buffer.len().next_multiple_of(576), 0f32);
            for chunk in buffer.drain(..).as_slice().chunks_exact(576) {
                let offset = output.len();
                let frames = resampler.output_frames_max();
                let newcap = offset + frames;
                output.resize(newcap, 0f32);
                let (_, written) = try_harder!(resampler
                    .process_partial_into_buffer(Some(&[chunk]), &mut [&mut output[offset..]], None)
                    .map(Some)
                    .transpose());
                output.truncate(offset + written);
            }
            output.drain(..)
        } else {
            self.buffer.drain(..)
        };
        Some(Ok(buffer))
    }
}

fn _decode(ext: &str, istrm: MediaSourceStream) -> Result<Vec<f32>> {
    let fmt: Box<dyn FormatReader> = match ext {
        "mp3" | "727" => Box::new(MpaReader::try_new(istrm, &Default::default())?),
        "ogg" => Box::new(OggReader::try_new(istrm, &Default::default())?),
        _ => {
            let probe = symphonia::default::get_probe();
            let hint = std::mem::take(Hint::new().with_extension(ext));
            probe
                .format(
                    &hint,
                    istrm,
                    &Default::default(),
                    &MetadataOptions {
                        limit_metadata_bytes: Limit::None,
                        ..Default::default()
                    },
                )?
                .format
        }
    };
    let track = fmt
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .map(Ok)
        .unwrap_or_else(|| Err(ErrorKind::UnsupportedCodec))?;
    let track_id = track.id;

    let sample_count = track
        .codec_params
        .n_frames
        .unwrap_or(1 << 24)
        // .map(Ok)
        // .unwrap_or_else(|| Err(Error::from_kind(ErrorKind::UnknownFrames)))?
        as usize;
    let sample_rate = track
        .codec_params
        .sample_rate
        .map(Ok)
        .unwrap_or_else(|| Err(Error::from_kind(ErrorKind::UnknownRate)))?;
    let adjusted_sample_count = (sample_count * 48000).div_ceil(sample_rate as usize);
    let mut output = Vec::with_capacity(adjusted_sample_count);
    let mut stream =
        MonoStream::new(fmt, track_id).with_resample_ratio(48000.0 / sample_rate as f64);
    while let Some(payload) = stream.consume_next() {
        output.extend_from_slice(payload?.as_slice());
    }
    // let samples_found = output.len();
    // if adjusted_sample_count
    //     .checked_sub(samples_found)
    //     .unwrap_or(0)
    //     > 576
    // {
    //     Err(Error::from_kind(ErrorKind::Py(PyValueError::new_err(
    //         format!("end of stream: {samples_found} samples found; {adjusted_sample_count} samples expected"),
    //     ))))
    // } else {
    //     Ok(output)
    // }
    Ok(output)
}

fn write_hit_json(
    src: &HitObject,
    meta: &MetadataSection,
    difficulty: &DifficultySection,
    timing: &TimingPoint,
    dst: &mut String,
) {
    let &TimingPoint {
        kiai_mode,
        meter,
        ms_per_beat: beat_length,
        ..
    } = &timing;
    let &MetadataSection {
        beatmap_id,
        beatmap_set_id,
        ..
    } = meta;
    let &DifficultySection {
        hp_drain_rate,
        circle_size,
        overall_difficulty,
        approach_rate,
        slider_multiplier,
        slider_tick_rate,
    } = difficulty;
    let write_circle = |dst: &mut JSONObjectWriter,
                        HitCircle {
                            x,
                            y,
                            new_combo,
                            time,
                            color_skip,
                            hitsound,
                            ..
                        }| {
        dst.value("beatmap_id", beatmap_id);
        dst.value("beatmap_set_id", beatmap_set_id);
        dst.value("start_time", time);
        dst.value("new_combo", new_combo);
        dst.value("color_skip", color_skip);
        dst.value("hitsound", hitsound);
        {
            let mut position = dst.object("position");
            position.value("x", x);
            position.value("y", y);
        }
        {
            let mut difficulty = dst.object("difficulty");
            difficulty.value("hp_drain_rate", hp_drain_rate);
            difficulty.value("circle_size", circle_size);
            difficulty.value("overall_difficulty", overall_difficulty);
            difficulty.value("approach_rate", approach_rate);
            difficulty.value("slider_multiplier", slider_multiplier);
            difficulty.value("slider_tick_rate", slider_tick_rate);
        }
        {
            let mut timing = dst.object("timing");
            timing.value("kiai_mode", kiai_mode);
            timing.value("meter", meter);
            timing.value("beat_length", beat_length);
        }
    };
    match src {
        &HitObject::HitCircle(ref circle) => {
            {
                let mut dst = JSONObjectWriter::new(dst);
                write_circle(
                    &mut dst,
                    HitCircle {
                        extras: Default::default(),
                        ..*circle
                    },
                );
                dst.value("ctl_index", 0);
                dst.value("slider_type", "n/a");
                dst.value("repeat", 0);
                dst.value("hit_type", "hit");
                dst.value("end_time", -1);
            }
            let _ = dst.write_str("\n");
        }
        &HitObject::Slider(Slider {
            x,
            y,
            new_combo,
            color_skip,
            time,
            ref slider_type,
            ref curve_points,
            hitsound,
            repeat,
            ..
        }) => {
            let circle = HitCircle {
                x,
                y,
                new_combo,
                color_skip,
                time,
                hitsound: hitsound,
                extras: Default::default(),
            };
            for (x, y) in std::iter::once((x, y)).chain(curve_points.iter().copied()) {
                let circle = HitCircle {
                    x,
                    y,
                    extras: Default::default(),
                    ..circle
                };
                {
                    let mut dst = JSONObjectWriter::new(dst);
                    write_circle(&mut dst, circle);
                    dst.value("ctl_index", 0);
                    dst.value(
                        "slider_type",
                        match slider_type {
                            SliderType::Bezier => "bezier",
                            SliderType::Catmull => "catmull",
                            SliderType::Perfect => "circle",
                            SliderType::Linear => "linear",
                        },
                    );
                    dst.value("repeat", repeat);
                    dst.value("hit_type", "slider");
                    dst.value("end_time", -1);
                }
                let _ = dst.write_str("\n");
            }
        }
        &HitObject::Spinner(Spinner {
            x,
            y,
            new_combo,
            color_skip,
            time,
            hitsound,
            end_time,
            ..
        }) => {
            {
                let mut dst = JSONObjectWriter::new(dst);
                write_circle(
                    &mut dst,
                    HitCircle {
                        x,
                        y,
                        new_combo,
                        color_skip,
                        time,
                        hitsound,
                        extras: Default::default(),
                    },
                );
                dst.value("ctl_index", 0);
                dst.value("hit_type", "spinner");
                dst.value("end_time", end_time);
                dst.value("slider_type", "n/a");
                dst.value("repeat", json_writer::NULL);
            }
            let _ = dst.write_str("\n");
        }
        &HitObject::HoldNote(HoldNote {
            x,
            y,
            new_combo,
            color_skip,
            time,
            hitsound,
            end_time,
            ..
        }) => {
            {
                let mut dst = JSONObjectWriter::new(dst);
                write_circle(
                    &mut dst,
                    HitCircle {
                        x,
                        y,
                        new_combo,
                        color_skip,
                        time,
                        hitsound,
                        extras: Default::default(),
                    },
                );
                dst.value("ctl_index", 0);
                dst.value("hit_type", "held");
                dst.value("end_time", end_time);
                dst.value("slider_type", "n/a");
                dst.value("repeat", json_writer::NULL);
            }
            let _ = dst.write_str("\n");
        }
    }
}

#[pyfunction]
fn extract<'py>(py: Python<'py>, path: &str) -> PyResult<(&'py PyArray1<f32>, String)> {
    let (samples, hits_json) = py
        .allow_threads(|| {
            let file = std::fs::File::open(path)?;
            let mut archive = zip::ZipArchive::new(file)?;
            let mut hit_json_string = String::with_capacity(1 << 20);
            let mut audio_path = None;
            let beatmap_names = archive
                .file_names()
                .filter(|name| name.ends_with(".osu"))
                .map(String::from)
                .collect::<Vec<_>>();
            if beatmap_names.len() == 0 {
                return Err(PyValueError::new_err("no .osu files in archive").into());
            }
            {
                // let mut hit_records = json_writer::JSONArrayWriter::new(&mut hit_json_string);
                for name in beatmap_names {
                    let mut entry = archive.by_name(&name)?;
                    let mut data = Vec::with_capacity(entry.size() as usize);
                    std::io::copy(&mut entry, &mut data)?;
                    let mut beatmap = osuparse::parse_beatmap(std::str::from_utf8(&data)?)
                        .map_err(|err| {
                            Error::from_kind(ErrorKind::Py(PyValueError::new_err(format!(
                                "parse beatmap '{path}/{name}': {err}"
                            ))))
                        })?;
                    audio_path =
                        Some(audio_path.unwrap_or_else(|| beatmap.general.audio_filename.clone()));
                    if audio_path.as_ref() != Some(&beatmap.general.audio_filename) {
                        let prev = audio_path.unwrap();
                        let next = beatmap.general.audio_filename.as_str();
                        return Err(Error::from_kind(ErrorKind::Py(
                            PyNotImplementedError::new_err(format!(
                                "mismatched beatmap audio paths {prev} and {next}"
                            )),
                        )));
                    }

                    beatmap.timing_points.sort_unstable_by(
                        |&TimingPoint { offset: a, .. }, &TimingPoint { offset: b, .. }| {
                            a.total_cmp(&b)
                        },
                    );
                    let mut timings = beatmap.timing_points.into_iter().peekable();
                    timings
                        .peek()
                        .map(|point| {
                            if point.inherited {
                                Ok(())
                            } else {
                                Err(Error::from(PyValueError::new_err(format!(
                                    "parse {path}/{name}: first timing point inherited"
                                ))))
                            }
                        })
                        .unwrap_or_else(|| {
                            Err(Error::from(PyValueError::new_err(format!(
                                "parse {path}/{name}: first timing point missing"
                            ))))
                        })?;

                    let mut carry_timing = timings.next().unwrap();
                    let hits = beatmap.hit_objects.iter().peekable();
                    for hit in hits {
                        let start_time = *match hit {
                            HitObject::HitCircle(HitCircle { time, .. }) => time,
                            HitObject::HoldNote(HoldNote { time, .. }) => time,
                            HitObject::Spinner(Spinner { time, .. }) => time,
                            HitObject::Slider(Slider { time, .. }) => time,
                        } as f32;

                        {
                            carry_timing = if timings
                                .peek()
                                .filter(|&TimingPoint { offset, .. }| offset < &start_time)
                                .is_some()
                            {
                                timings.next().unwrap()
                            } else {
                                carry_timing
                            };

                            write_hit_json(
                                hit,
                                &beatmap.metadata,
                                &beatmap.difficulty,
                                &carry_timing,
                                &mut hit_json_string,
                            );
                        }
                    }
                }
            }
            let samples = {
                let audio_path = audio_path.unwrap();
                let mut entry = archive
                    .by_name(&audio_path)
                    .chain_err(|| ErrorKind::MissingAudio(audio_path.clone()))?;
                let mut data = Vec::with_capacity(entry.size() as usize);
                std::io::copy(&mut entry, &mut data)?;
                let cursor = std::io::Cursor::new(data);
                let boxed = Box::new(cursor);
                let istrm = MediaSourceStream::new(boxed, Default::default());
                let lower = audio_path.to_lowercase();
                let ext = &lower[lower.len() - 3..];
                _decode(ext, istrm)
            }?;
            Ok::<_, Error>((samples, hit_json_string))
        })
        .map_err(Into::<PyErr>::into)?;
    let samples = samples.into_pyarray(py);
    Ok((samples, hits_json))
}

#[pyfunction]
fn decode<'py>(py: Python<'py>, data: Vec<u8>) -> PyResult<&'py PyArray1<f32>> {
    let samples = py
        .allow_threads(|| {
            let istrm = {
                let cursor = std::io::Cursor::new(data);
                let boxed = Box::new(cursor);
                MediaSourceStream::new(boxed, Default::default())
            };
            _decode("", istrm)
        })
        .map_err(Into::<PyErr>::into)?;
    let samples = samples.into_pyarray(py);
    Ok(samples)
}
/// A Python module implemented in Rust.
#[pymodule]
fn audio_decode(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decode, m)?)?;
    m.add_function(wrap_pyfunction!(extract, m)?)?;
    Ok(())
}
