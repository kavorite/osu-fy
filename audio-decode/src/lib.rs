use numpy::{IntoPyArray, PyArray1};
use pyo3::{
    exceptions::{PyNotImplementedError, PyValueError},
    prelude::*,
};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use symphonia::core::{
    audio::{AudioBuffer, AudioBufferRef},
    codecs::{DecoderOptions, CODEC_TYPE_NULL},
    errors::Error as SymphoniaError,
    formats::FormatReader,
    io::MediaSourceStream,
    meta::{Limit, MetadataOptions},
    probe::Hint,
    sample::Sample,
};
use symphonia::default::formats::{MpaReader, OggReader};

use error_chain::error_chain;

error_chain! {
    errors {
        NoAudioFilesFound {
            display("a supported audio file extension could not be found")
        }
        UnsupportedCodec {
            display("a supported audio track could not be found")
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
    }
}

impl Into<PyErr> for Error {
    fn into(self) -> PyErr {
        match self.kind() {
            ErrorKind::UnsupportedCodec => PyNotImplementedError::new_err(self.to_string()),
            ErrorKind::UnsupportedSampleFormat(_) => {
                PyNotImplementedError::new_err(self.to_string())
            }
            ErrorKind::Decode(err) => PyValueError::new_err(format!("decode: {err}")),
            ErrorKind::Resample(err) => PyValueError::new_err(format!("resample: {err}")),
            _ => PyValueError::new_err(self.to_string()),
        }
    }
}

fn load_mono_buf<S>(src: impl AsRef<AudioBuffer<S>>, dst: &mut [f32])
where
    S: Sample + num::Zero,
    f64: From<S>,
{
    let planes = src.as_ref().planes();
    let scale = if S::zero() == S::MID {
        1 << (S::EFF_BITS - 1)
    } else {
        1 << S::EFF_BITS
    } as f64;
    let shift: f64 = S::MID.into();
    for plane in planes.planes() {
        for (i, x) in plane.iter().copied().enumerate() {
            let value: f64 = x.into();
            dst[i] += ((value - shift) / scale) as f32;
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
            let planes = src.as_ref().planes();
            for plane in planes.planes() {
                for (i, value) in plane.iter().copied().enumerate() {
                    let value: f64 = value.0.into();
                    let scale = (1 << 23) as f64;
                    dst[i] += (value / scale) as f32;
                }
            }
        }
        S32(src) => load_mono_buf(src, dst),
        U8(src) => load_mono_buf(src, dst),
        U16(src) => load_mono_buf(src, dst),
        U24(src) => {
            let planes = src.as_ref().planes();
            let shift: f64 = (1 << 23) as f64;
            let scale = (1 << 24) as f64;
            for plane in planes.planes() {
                for (i, value) in plane.iter().copied().enumerate() {
                    let value: f64 = value.0.into();
                    dst[i] += ((value - shift) / scale) as f32;
                }
            }
        }
        U32(src) => load_mono_buf(src, dst),
    }
}

fn _decode(ext: &str, istrm: MediaSourceStream) -> Result<Vec<f32>> {
    let mut fmt: Box<dyn FormatReader> = match ext {
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
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions { verify: false })?;
    let mut output = Vec::with_capacity(track.codec_params.n_frames.unwrap_or(0) as usize);
    let mut buffer = Vec::new();

    let mut resampler = None;

    loop {
        let packet = match fmt.next_packet() {
            Ok(packet) => Ok(packet),
            Err(SymphoniaError::IoError(err))
                if err.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(SymphoniaError::DecodeError(_) | SymphoniaError::IoError(_)) => continue,
            Err(err) => Err(err),
        }?;
        if packet.track_id() != track_id {
            continue;
        }

        let packet = decoder.decode(&packet)?;
        let packet_spec = packet.spec();
        if resampler.is_none() && packet_spec.rate != 48000 {
            let params = SincInterpolationParameters {
                sinc_len: 256,
                f_cutoff: 0.95,
                interpolation: SincInterpolationType::Linear,
                oversampling_factor: 256,
                window: WindowFunction::BlackmanHarris2,
            };
            resampler = Some(
                SincFixedIn::<f32>::new(48000.0 / (packet_spec.rate as f64), 2.0, params, 576, 1)
                    .unwrap(),
            );
        }

        let frames = packet.frames();
        buffer.resize(frames + buffer.len(), 0f32);
        load_mono(&packet, &mut buffer);
        if buffer.len() < 576 {
            continue;
        }
        if let Some(ref mut resampler) = resampler {
            let mut samples = resampler
                .process(&[buffer.as_slice()], None)?
                .into_iter()
                .next()
                .unwrap_or_else(Vec::new);
            buffer.truncate(0);
            output.append(&mut samples);
        } else {
            output.append(&mut buffer);
        }
    }
    Ok(output)
}

#[pyfunction]
fn extract<'py>(py: Python<'py>, path: &str) -> PyResult<&'py PyArray1<f32>> {
    let samples = py
        .allow_threads(|| {
            let file = std::fs::File::open(path)?;
            let mut archive = zip::ZipArchive::new(file)?;
            let name = archive
                .file_names()
                .find(|name| {
                    let name = name.to_lowercase();
                    name.ends_with(".mp3") || name.ends_with(".ogg") || name.ends_with(".727")
                })
                .map(String::from)
                .map(Ok)
                .unwrap_or_else(|| Err(ErrorKind::NoAudioFilesFound))?;
            let mut entry = archive.by_name(name.as_str())?;
            let mut data = Vec::with_capacity(entry.size() as usize);
            std::io::copy(&mut entry, &mut data)?;
            let cursor = std::io::Cursor::new(data);
            let boxed = Box::new(cursor);
            let istrm = MediaSourceStream::new(boxed, Default::default());
            let lower = name.to_lowercase();
            let ext = &lower[lower.len() - 3..];
            _decode(ext, istrm)
        })
        .map_err(Into::<PyErr>::into)?;
    let samples = samples.into_pyarray(py);
    Ok(samples)
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
