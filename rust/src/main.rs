#![feature(portable_simd)]
#![feature(test)]
#![feature(byte_slice_trim_ascii)]
#![feature(slice_split_once)]

#[macro_use]
extern crate lazy_static;

extern crate test;

use memmap::{Mmap, MmapOptions};
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::{
    collections::HashMap,
    env,
    fs::File,
    io::{BufRead, BufReader, Read},
    simd::{num::SimdInt, Simd},
};

#[derive(Debug, Copy, Clone)]
struct MeasurementStats {
    min: f32,
    max: f32,
    sum: f32,
    count: usize,
}

impl MeasurementStats {
    fn new(val: f32) -> Self {
        MeasurementStats {
            min: val,
            max: val,
            sum: val,
            count: 1,
        }
    }

    fn add(&self, val: f32) -> Self {
        MeasurementStats {
            min: f32::min(self.min, val),
            max: f32::max(self.max, val),
            sum: self.sum + val,
            count: self.count + 1,
        }
    }

    fn add_mut(&mut self, val: f32) {
        self.min = f32::min(self.min, val);
        self.max = f32::max(self.max, val);
        self.sum += val;
        self.count += 1;
    }

    fn merge_mut(&mut self, right: &Self) {
        self.min = f32::min(self.min, right.min);
        self.max = f32::max(self.max, right.max);
        self.sum = self.sum + right.sum;
        self.count = self.count + right.count;
    }
}

#[derive(Debug, Copy, Clone)]
struct MeasurementStatsInt {
    min: i32,
    max: i32,
    sum: i32,
    count: usize,
}

impl MeasurementStatsInt {
    fn new(val: i32) -> Self {
        MeasurementStatsInt {
            min: val,
            max: val,
            sum: val,
            count: 1,
        }
    }

    fn add(&self, val: i32) -> Self {
        MeasurementStatsInt {
            min: i32::min(self.min, val),
            max: i32::max(self.max, val),
            sum: self.sum + val,
            count: self.count + 1,
        }
    }

    fn add_mut(&mut self, val: i32) {
        self.min = i32::min(self.min, val);
        self.max = i32::max(self.max, val);
        self.sum += val;
        self.count += 1;
    }

    fn merge_mut(&mut self, right: &Self) {
        self.min = i32::min(self.min, right.min);
        self.max = i32::max(self.max, right.max);
        self.sum = self.sum + right.sum;
        self.count = self.count + right.count;
    }
}

#[derive(Debug, Copy, Clone)]
struct MeasurementStatsMask {
    min_max: i32,
    sum_count: i64,
}

impl MeasurementStatsMask {
    fn new(val: f32) -> Self {
        let v = (val * 10.0) as i32;

        MeasurementStatsMask {
            min_max: (v << 16) & v as i32,
            sum_count: ((v as i64) << 32) & 1i64,
        }
    }

    fn add(&self, val: f32) -> Self {
        let v = (val * 10.0) as i32;
        let min = i32::min(self.min_max >> 16, v);
        let max = i32::max((self.min_max << 16) >> 16, v);
        let sum = (self.sum_count >> 32) + v as i64;
        let count = ((self.sum_count << 32) >> 32) + 1;

        MeasurementStatsMask {
            min_max: (min << 16) & max,
            sum_count: (sum << 32) & count,
        }
    }

    fn add_mut(&mut self, val: f32) {
        let v = (val * 10.0) as i32;
        let min = i32::min(self.min_max >> 16, v);
        let max = i32::max((self.min_max << 16) >> 16, v);
        let sum = (self.sum_count >> 32) + v as i64;
        let count = ((self.sum_count << 32) >> 32) + 1;
        self.min_max = (min << 16) & max;
        self.sum_count = (sum << 32) & count;
    }

    fn merge_mut(&mut self, right: &Self) {
        let min = i32::min(self.min_max >> 16, right.min_max >> 16);
        let max = i32::max(self.min_max << 16, right.min_max << 16);
        self.min_max = (min << 16) & max;
        let sum = (self.sum_count >> 32) + (right.sum_count >> 32);
        let count = ((self.sum_count << 32) >> 32) + ((right.sum_count << 32) >> 32);
        self.sum_count = (sum << 32) & count;
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let file = "../_data/measurements_1b.txt";
    if args.len() == 1 {
        println!(
            "{}",
            handle_fhash_dint_read_until_mut_cparse_par("../_data/measurements_1b.txt")
        )
    }
    let res = match args[1].as_str() {
        "00" => handle_naive(file),
        "01" => handle_fhash(file),
        "02" => handle_fhash_ffloat(file),
        "03" => handle_fhash_ffloat_read_until(file),
        "04" => handle_fhash_lookup_read_until(file),
        "05" => handle_fhash_ffloat_read_until_mut(file),
        "06" => handle_fhash_dfloat_read_until_mut(file),
        "07" => handle_fhash_dint_read_until_mut(file),
        "08" => handle_fhash_dint_read_until_mut_memchr(file),
        "09" => handle_fhash_dint_read_until_mut_memchr_cparse(file),
        "10" => handle_fhash_dint_read_until_mut_cparse_par(file),
        _ => handle_fhash_dint_read_until_mut_cparse_par(file),
    };
    let res = handle_fhash_dint_read_until_mut_cparse_par("../_data/measurements_1b.txt");
    // let res = handle_fhash_dint_read_until_mut_memchr_cparse("../_data/measurements_1b.txt");
    println!("{}", res)
}

fn handle_naive(file: &str) -> String {
    let file = File::open(file).unwrap();
    let mut m: HashMap<String, MeasurementStats> = HashMap::new();
    let reader = BufReader::with_capacity(8096, file);
    reader.lines().for_each(|line| {
        let l = line.unwrap();
        let (city, measurement_str) = l.split_once(";").unwrap();
        let measurement = measurement_str.parse::<f32>().unwrap();
        if let Some(existing) = m.get(city) {
            m.insert(city.to_string(), existing.add(measurement));
        } else {
            m.insert(city.to_owned(), MeasurementStats::new(measurement));
        }
    });

    let mut results = m
        .iter()
        .map(|(city, measurement)| {
            format!(
                "{}={:.1}/{:.1}/{:.1}",
                city,
                measurement.min,
                measurement.sum / measurement.count as f32,
                measurement.max
            )
        })
        .collect::<Vec<_>>();
    results.sort();
    format!("{{{}}}", results.join(", "))
}

fn handle_fhash(file: &str) -> String {
    let file = File::open(file).unwrap();
    let mut m: FxHashMap<String, MeasurementStats> = FxHashMap::default();
    let reader = BufReader::with_capacity(8096, file);
    reader.lines().for_each(|line| {
        let l = line.unwrap();
        let (city, measurement_str) = l.split_once(";").unwrap();
        let measurement = measurement_str.parse::<f32>().unwrap();
        if let Some(existing) = m.get(city) {
            m.insert(city.to_string(), existing.add(measurement));
        } else {
            m.insert(city.to_owned(), MeasurementStats::new(measurement));
        }
    });

    let mut results = m
        .iter()
        .map(|(city, measurement)| {
            format!(
                "{}={:.1}/{:.1}/{:.1}",
                city,
                measurement.min,
                measurement.sum / measurement.count as f32,
                measurement.max
            )
        })
        .collect::<Vec<_>>();
    results.sort();
    format!("{{{}}}", results.join(", "))
}

fn handle_fhash_ffloat(file: &str) -> String {
    let file = File::open(file).unwrap();
    let mut m: FxHashMap<String, MeasurementStats> = FxHashMap::default();
    let reader = BufReader::with_capacity(8096, file);
    reader.lines().for_each(|line| {
        let l = line.unwrap();
        let (city, measurement_str) = l.split_once(";").unwrap();
        let measurement = fast_float::parse(measurement_str).unwrap();
        if let Some(existing) = m.get(city) {
            m.insert(city.to_string(), existing.add(measurement));
        } else {
            m.insert(city.to_owned(), MeasurementStats::new(measurement));
        }
    });

    let mut results = m
        .iter()
        .map(|(city, measurement)| {
            format!(
                "{}={:.1}/{:.1}/{:.1}",
                city,
                measurement.min,
                measurement.sum / measurement.count as f32,
                measurement.max
            )
        })
        .collect::<Vec<_>>();
    results.sort();
    format!("{{{}}}", results.join(", "))
}

fn handle_fhash_ffloat_read_until(file: &str) -> String {
    let file = File::open(file).unwrap();
    let mut m: FxHashMap<Vec<u8>, MeasurementStats> = FxHashMap::default();
    let mut reader = BufReader::new(file);
    let mut buf = Vec::with_capacity(256);
    while let Ok(n) = reader.read_until(b'\n', &mut buf) {
        if n == 0 {
            break;
        }
        let s = &buf[..n - 1];
        if let Some((city, measurement_str)) = s.split_once(|&c| c == b';') {
            let measurement = fast_float::parse(measurement_str).unwrap();
            if let Some(existing) = m.get(city) {
                m.insert(city.to_owned(), existing.add(measurement));
            } else {
                m.insert(city.to_owned(), MeasurementStats::new(measurement));
            }
        }
        buf.clear();
    }

    let mut results = m
        .iter()
        .map(|(city, measurement)| {
            format!(
                "{}={:.1}/{:.1}/{:.1}",
                String::from_utf8(city.to_owned()).unwrap(),
                measurement.min,
                measurement.sum / measurement.count as f32,
                measurement.max
            )
        })
        .collect::<Vec<_>>();
    results.sort();
    format!("{{{}}}", results.join(", "))
}

lazy_static! {
    static ref HASHMAP: FxHashMap<Vec<u8>, f32> = {
        let mut m = FxHashMap::default();
        for d in -99..=99 {
            for f in 0..=9 {
                let n = d as f32 + (0.1 * f as f32);
                let k = format!("{:.1}", n);
                m.insert(k.as_bytes().to_vec(), n);
            }
        }
        m
    };
}

fn handle_fhash_lookup_read_until(file: &str) -> String {
    let file = File::open(file).unwrap();
    let mut m: FxHashMap<Vec<u8>, MeasurementStats> = FxHashMap::default();
    let mut reader = BufReader::new(file);
    let mut buf = Vec::with_capacity(256);
    while let Ok(n) = reader.read_until(b'\n', &mut buf) {
        if n == 0 {
            break;
        }
        let s = &buf[..n - 1];
        if let Some((city, measurement_str)) = s.split_once(|&c| c == b';') {
            let measurement = *HASHMAP
                .get(measurement_str)
                .ok_or_else(|| {
                    println!(
                        "value not found: {}",
                        String::from_utf8(measurement_str.to_vec()).unwrap()
                    )
                })
                .unwrap();
            if let Some(existing) = m.get(city) {
                m.insert(city.to_owned(), existing.add(measurement));
            } else {
                m.insert(city.to_owned(), MeasurementStats::new(measurement));
            }
        }
        buf.clear();
    }

    let mut results = m
        .iter()
        .map(|(city, measurement)| {
            format!(
                "{}={:.1}/{:.1}/{:.1}",
                String::from_utf8(city.to_owned()).unwrap(),
                measurement.min,
                measurement.sum / measurement.count as f32,
                measurement.max
            )
        })
        .collect::<Vec<_>>();
    results.sort();
    format!("{{{}}}", results.join(", "))
}

fn handle_fhash_ffloat_read_until_mut(file: &str) -> String {
    let file = File::open(file).unwrap();
    let mut m: FxHashMap<Vec<u8>, MeasurementStats> = FxHashMap::default();
    let mut reader = BufReader::new(file);
    let mut buf = Vec::with_capacity(256);
    while let Ok(n) = reader.read_until(b'\n', &mut buf) {
        if n == 0 {
            break;
        }
        let s = &buf[..n - 1];
        if let Some((city, measurement_str)) = s.split_once(|&c| c == b';') {
            let measurement = fast_float::parse(measurement_str).unwrap();
            if let Some(existing) = m.get_mut(city) {
                existing.add_mut(measurement);
            } else {
                m.insert(city.to_owned(), MeasurementStats::new(measurement));
            }
        }
        buf.clear();
    }

    let mut results = m
        .iter()
        .map(|(city, measurement)| {
            format!(
                "{}={:.1}/{:.1}/{:.1}",
                String::from_utf8(city.to_owned()).unwrap(),
                measurement.min,
                measurement.sum / measurement.count as f32,
                measurement.max
            )
        })
        .collect::<Vec<_>>();
    results.sort();
    format!("{{{}}}", results.join(", "))
}

/**
 * Dum dum parsing function to make use of the knowledge that
 * the values are all between -99.9 and 99.9.
 */
#[inline(always)]
fn dum_parse(bs: &[u8]) -> f32 {
    let mut idx = bs.len();
    let sign_factor = match bs[0] {
        b'-' => -0.1,
        _ => 0.1,
    };
    let stop_idx = if sign_factor > 0.0 { 0 } else { 1 };
    let factors = Simd::from_array([0, 100, 10, 1]);
    let mut components: Simd<i32, 4> = Simd::splat(0i32);
    let mut simd_idx = 3usize;
    while idx > stop_idx {
        let c = bs[idx - 1];
        if c == b'.' {
            idx -= 1;
            continue;
        }
        components[simd_idx] = (c - b'0') as i32;
        idx -= 1;
        simd_idx -= 1;
    }
    ((factors * components).reduce_sum() as f32) * sign_factor
}

/**
 * Dum dum parsing function to make use of the knowledge that
 * the values are all between -99.9 and 99.9.
 */
#[inline(always)]
fn dum_parse_as_i32(bs: &[u8]) -> i32 {
    let mut idx = bs.len();
    let sign_factor = match bs[0] {
        b'-' => -1,
        _ => 1,
    };
    let stop_idx = if sign_factor > 0 { 0 } else { 1 };
    let factors = Simd::from_array([0, 100, 10, 1]);
    let mut components: Simd<i32, 4> = Simd::splat(0i32);
    let mut simd_idx = 3usize;
    while idx > stop_idx {
        let c = bs[idx - 1];
        if c == b'.' {
            idx -= 1;
            continue;
        }
        components[simd_idx] = (c - b'0') as i32;
        idx -= 1;
        simd_idx -= 1;
    }
    ((factors * components).reduce_sum()) * sign_factor
}

fn handle_fhash_dfloat_read_until_mut(file: &str) -> String {
    let file = File::open(file).unwrap();
    let mut m: FxHashMap<Vec<u8>, MeasurementStats> = FxHashMap::default();
    let mut reader = BufReader::new(file);
    let mut buf = Vec::with_capacity(256);
    while let Ok(n) = reader.read_until(b'\n', &mut buf) {
        if n == 0 {
            break;
        }
        let s = &buf[..n - 1];
        if let Some((city, measurement_str)) = s.split_once(|&c| c == b';') {
            let measurement = dum_parse(measurement_str);
            if let Some(existing) = m.get_mut(city) {
                existing.add_mut(measurement);
            } else {
                m.insert(city.to_owned(), MeasurementStats::new(measurement));
            }
        }
        buf.clear();
    }

    let mut results = m
        .iter()
        .map(|(city, measurement)| {
            format!(
                "{}={:.1}/{:.1}/{:.1}",
                String::from_utf8(city.to_owned()).unwrap(),
                measurement.min,
                measurement.sum / measurement.count as f32,
                measurement.max
            )
        })
        .collect::<Vec<_>>();
    results.sort();
    format!("{{{}}}", results.join(", "))
}

fn handle_fhash_dint_read_until_mut(file: &str) -> String {
    let file = File::open(file).unwrap();
    let mut m: FxHashMap<Vec<u8>, MeasurementStatsInt> = FxHashMap::default();
    let mut reader = BufReader::new(file);
    let mut buf = Vec::with_capacity(64);
    while let Ok(n) = reader.read_until(b'\n', &mut buf) {
        if n == 0 {
            break;
        }
        let s = &buf[..n - 1];
        if let Some((city, measurement_str)) = s.split_once(|&c| c == b';') {
            let measurement = dum_parse_as_i32(measurement_str);
            if let Some(existing) = m.get_mut(city) {
                existing.add_mut(measurement);
            } else {
                m.insert(city.to_owned(), MeasurementStatsInt::new(measurement));
            }
        }
        buf.clear();
    }

    let mut results = m
        .iter()
        .map(|(city, measurement)| {
            format!(
                "{}={:.1}/{:.1}/{:.1}",
                String::from_utf8(city.to_owned()).unwrap(),
                measurement.min as f32 * 0.1,
                (measurement.sum as f32 * 0.1) / measurement.count as f32,
                measurement.max as f32 * 0.1
            )
        })
        .collect::<Vec<_>>();
    results.sort();
    format!("{{{}}}", results.join(", "))
}

fn handle_fhash_dint_read_until_mut_memchr(file: &str) -> String {
    let file = File::open(file).unwrap();
    let mut m: FxHashMap<Vec<u8>, MeasurementStatsInt> = FxHashMap::default();
    let mut reader = BufReader::new(file);
    let mut buf = Vec::with_capacity(64);
    while let Ok(n) = reader.read_until(b'\n', &mut buf) {
        if n == 0 {
            break;
        }
        let s = &buf[..n - 1];
        let pos = memchr::memchr(b';', s).unwrap();
        let city = &s[..pos];
        let measurement_str = &s[pos + 1..];
        let measurement = dum_parse_as_i32(measurement_str);
        if let Some(existing) = m.get_mut(city) {
            existing.add_mut(measurement);
        } else {
            m.insert(city.to_owned(), MeasurementStatsInt::new(measurement));
        }
        buf.clear();
    }

    let mut results = m
        .iter()
        .map(|(city, measurement)| {
            format!(
                "{}={:.1}/{:.1}/{:.1}",
                String::from_utf8(city.to_owned()).unwrap(),
                measurement.min as f32 * 0.1,
                (measurement.sum as f32 * 0.1) / measurement.count as f32,
                measurement.max as f32 * 0.1
            )
        })
        .collect::<Vec<_>>();
    results.sort();
    format!("{{{}}}", results.join(", "))
}

fn parse<R: Read>(r: &mut R) -> FxHashMap<Vec<u8>, MeasurementStatsInt> {
    let mut read = 0;
    let mut buf = [0u8; 4096];
    let mut m: FxHashMap<Vec<u8>, MeasurementStatsInt> = FxHashMap::default();
    let mut spillover: Vec<u8> = Vec::with_capacity(128);
    loop {
        let mut curr = 0;
        read = r.read(&mut buf).unwrap();
        if read == 0 {
            break;
        }
        if !spillover.is_empty() {
            match memchr::memchr(b'\n', &buf[curr..read]) {
                Some(p) => {
                    spillover.extend_from_slice(&buf[curr..p]);
                    let s = &spillover.as_slice();
                    let pos = memchr::memchr(b';', s).unwrap();
                    let city = &s[..pos];
                    let measurement_str = &s[pos + 1..];
                    let measurement = dum_parse_as_i32(measurement_str);
                    if let Some(existing) = m.get_mut(city) {
                        existing.add_mut(measurement);
                    } else {
                        m.insert(city.to_owned(), MeasurementStatsInt::new(measurement));
                    }
                    spillover.clear();
                    curr += p + 1;
                }
                None => break,
            }
        }

        while let Some(p) = memchr::memchr(b'\n', &buf[curr..read]) {
            let s = &buf[curr..curr + p];
            let pos = memchr::memchr(b';', s).unwrap();
            let city = &s[..pos];
            let measurement_str = &s[pos + 1..];
            let measurement = dum_parse_as_i32(measurement_str);
            if let Some(existing) = m.get_mut(city) {
                existing.add_mut(measurement);
            } else {
                m.insert(city.to_owned(), MeasurementStatsInt::new(measurement));
            }
            curr += p + 1;
        }
        spillover.extend_from_slice(&buf[curr..]);
    }
    m
}

fn handle_fhash_dint_read_until_mut_memchr_cparse(file: &str) -> String {
    let file = File::open(file).unwrap();
    let mut reader = BufReader::new(file);
    let m = parse(&mut reader);
    let mut results = m
        .iter()
        .map(|(city, measurement)| {
            format!(
                "{}={:.1}/{:.1}/{:.1}",
                String::from_utf8(city.to_owned()).unwrap(),
                measurement.min as f32 * 0.1,
                (measurement.sum as f32 * 0.1) / measurement.count as f32,
                measurement.max as f32 * 0.1
            )
        })
        .collect::<Vec<_>>();
    results.sort();
    format!("{{{}}}", results.join(", "))
}

fn len_to_chunks(len: usize, chunk_count: usize) -> Vec<(usize, usize)> {
    let chunk_size = len / chunk_count;
    let mut v = Vec::new();
    for i in 0..chunk_count {
        v.push((i * chunk_size, (i * chunk_size) + chunk_size - 1))
    }
    for i in 1..chunk_count {
        assert_eq!(v.get(i - 1).unwrap().1, v.get(i).unwrap().0 - 1);
    }
    let (last_start, _) = v.pop().unwrap();
    v.push((last_start, len - 1)); // the last range end can be hardcoded to the known len-1
    assert_eq!(v.last().unwrap().1, len - 1);
    v
}

fn parse_chunk(mmap: &Mmap, range: (usize, usize)) -> FxHashMap<Vec<u8>, MeasurementStatsInt> {
    let (prelim_start, prelim_end) = range;
    let mut start = prelim_start;
    while *mmap.get(start).unwrap() != b'\n' {
        start += 1;
    }
    let mut end = prelim_end;
    while *mmap.get(end).unwrap() != b'\n' {
        end += 1;
    }
    let mut chunk = &mmap[start + 1..end];
    return parse(&mut chunk);
}

fn merge_map(
    left: &mut FxHashMap<Vec<u8>, MeasurementStatsInt>,
    right: FxHashMap<Vec<u8>, MeasurementStatsInt>,
) {
    right.iter().for_each(|(k, v)| match left.get_mut(k) {
        Some(measurements) => measurements.merge_mut(v),
        None => {
            left.insert(k.to_vec(), *v);
        }
    });
}

fn handle_fhash_dint_read_until_mut_cparse_par(file: &str) -> String {
    let file = File::open(file).unwrap();
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    let chunk_idxs = len_to_chunks(mmap.len(), 8);
    let m = chunk_idxs
        .par_iter()
        .map(|range| parse_chunk(&mmap, *range))
        .reduce(
            || FxHashMap::with_capacity_and_hasher(256, Default::default()),
            |mut left, right| {
                merge_map(&mut left, right);
                left
            },
        );

    let mut results = m
        .iter()
        .map(|(city, measurement)| {
            format!(
                "{}={:.1}/{:.1}/{:.1}",
                String::from_utf8(city.to_owned()).unwrap(),
                measurement.min as f32 * 0.1,
                (measurement.sum as f32 * 0.1) / measurement.count as f32,
                measurement.max as f32 * 0.1
            )
        })
        .collect::<Vec<_>>();
    results.sort();

    format!("{{{}}}", results.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[test]
    fn test_naive() {
        let expected = include_str!("../../_data/measurements_1m_result.txt").to_string();
        let actual = handle_naive("../_data/measurements_1m.txt");
        assert_eq!(expected, actual)
    }

    #[test]
    fn test_fhash() {
        let expected = include_str!("../../_data/measurements_1m_result.txt").to_string();
        let actual = handle_fhash("../_data/measurements_1m.txt");
        assert_eq!(expected, actual)
    }

    #[test]
    fn test_fhash_ffloat() {
        let expected = include_str!("../../_data/measurements_1m_result.txt").to_string();
        let actual = handle_fhash_ffloat("../_data/measurements_1m.txt");
        assert_eq!(expected, actual)
    }

    #[test]
    fn test_fhash_ffloat_read_until() {
        let expected = include_str!("../../_data/measurements_1m_result.txt").to_string();
        let actual = handle_fhash_ffloat_read_until("../_data/measurements_1m.txt");
        assert_eq!(expected, actual)
    }

    #[test]
    fn test_fhash_lookup_read_until() {
        let expected = include_str!("../../_data/measurements_1m_result.txt").to_string();
        let actual = handle_fhash_lookup_read_until("../_data/measurements_1m.txt");
        assert_eq!(expected, actual)
    }

    #[test]
    fn test_fhash_ffloat_read_until_mut() {
        let expected = include_str!("../../_data/measurements_1m_result.txt").to_string();
        let actual = handle_fhash_ffloat_read_until_mut("../_data/measurements_1m.txt");
        assert_eq!(expected, actual)
    }

    #[test]
    fn test_dum_parse() {
        let tests = [
            ("12.3", 12.3f32),
            ("1.2", 1.2f32),
            ("0.1", 0.1f32),
            ("0.0", 0.0f32),
            ("-0.1", -0.1f32),
            ("-1.2", -1.2f32),
            ("-12.3", -12.3f32),
        ];

        for (input, expected) in tests {
            assert_eq!(expected, dum_parse(input.as_bytes()))
        }
    }

    #[test]
    fn test_fhash_dfloat_read_until_mut() {
        let expected = include_str!("../../_data/measurements_1m_result.txt").to_string();
        let actual = handle_fhash_dfloat_read_until_mut("../_data/measurements_1m.txt");
        assert_eq!(expected, actual)
    }

    #[test]
    fn test_fhash_dint_read_until_mut() {
        let expected = include_str!("../../_data/measurements_1m_result.txt").to_string();
        let actual = handle_fhash_dint_read_until_mut("../_data/measurements_1m.txt");
        assert_eq!(expected, actual)
    }

    #[test]
    fn test_fhash_dint_read_until_mut_memchr() {
        let expected = include_str!("../../_data/measurements_1m_result.txt").to_string();
        let actual = handle_fhash_dint_read_until_mut_memchr("../_data/measurements_1m.txt");
        assert_eq!(expected, actual)
    }

    #[test]
    fn test_fhash_dfloat_lookup_read_until_mut_par() {
        let expected = include_str!("../../_data/measurements_1m_result.txt").to_string();
        let actual = handle_fhash_dint_read_until_mut_cparse_par("../_data/measurements_1m.txt");
        assert_eq!(expected, actual)
    }

    #[test]
    fn test_fhash_dint_read_until_mut_memchr_cparse() {
        let expected = include_str!("../../_data/measurements_1m_result.txt").to_string();
        let actual = handle_fhash_dint_read_until_mut_memchr_cparse("../_data/measurements_1m.txt");
        assert_eq!(expected, actual)
    }

    #[test]
    fn test_chunks() {
        let expected: Vec<(usize, usize)> = vec![(0, 3), (4, 7), (8, 11), (12, 15)];
        let actual = len_to_chunks(16, 4);
        assert_eq!(expected, actual)
    }

    #[bench]
    fn bench_00_naive(b: &mut Bencher) {
        b.iter(|| handle_naive("../_data/measurements_1m.txt"));
    }

    #[bench]
    fn bench_01_fhash(b: &mut Bencher) {
        b.iter(|| handle_fhash("../_data/measurements_1m.txt"));
    }

    #[bench]
    fn bench_02_fhash_ffloat(b: &mut Bencher) {
        b.iter(|| handle_fhash_ffloat("../_data/measurements_1m.txt"));
    }

    #[bench]
    fn bench_03_fhash_ffloat_read_until(b: &mut Bencher) {
        b.iter(|| handle_fhash_ffloat_read_until("../_data/measurements_1m.txt"));
    }

    #[bench]
    fn bench_04_fhash_lookup_read_until(b: &mut Bencher) {
        b.iter(|| handle_fhash_lookup_read_until("../_data/measurements_1m.txt"));
    }

    #[bench]
    fn bench_05_fhash_ffloat_read_until_mut(b: &mut Bencher) {
        b.iter(|| handle_fhash_ffloat_read_until_mut("../_data/measurements_1m.txt"));
    }

    #[bench]
    fn bench_06_fhash_dfloat_read_until_mut(b: &mut Bencher) {
        b.iter(|| handle_fhash_dfloat_read_until_mut("../_data/measurements_1m.txt"));
    }

    #[bench]
    fn bench_07_fhash_dint_read_until_mut(b: &mut Bencher) {
        b.iter(|| handle_fhash_dint_read_until_mut("../_data/measurements_1m.txt"));
    }

    #[bench]
    fn bench_08_fhash_dint_read_until_mut_memchr(b: &mut Bencher) {
        b.iter(|| handle_fhash_dint_read_until_mut_memchr("../_data/measurements_1m.txt"));
    }

    #[bench]
    fn bench_09_fhash_dint_read_until_mut_memchr_cparse(b: &mut Bencher) {
        b.iter(|| handle_fhash_dint_read_until_mut_memchr_cparse("../_data/measurements_1m.txt"));
    }

    #[bench]
    fn bench_10_fhash_dfloat_read_until_mut_memchr_cparse_par(b: &mut Bencher) {
        b.iter(|| handle_fhash_dint_read_until_mut_cparse_par("../_data/measurements_1m.txt"));
    }
}
