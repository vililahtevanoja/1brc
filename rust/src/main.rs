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
    fs::File,
    io::{BufRead, BufReader},
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

fn main() {
    let res = handle_fhash_ffloat_read_until_mut_par("../_data/measurements_1b.txt");
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

fn parse_chunk(mmap: &Mmap, range: (usize, usize)) -> FxHashMap<Vec<u8>, MeasurementStats> {
    let mut m: FxHashMap<Vec<u8>, MeasurementStats> = FxHashMap::default();
    let (prelim_start, prelim_end) = range;
    let mut start = prelim_start;
    while *mmap.get(start).unwrap() != b'\n' {
        start += 1;
    }
    let mut end = prelim_end;
    while *mmap.get(end).unwrap() != b'\n' {
        end += 1;
    }
    let chunk = &mmap[start..=end];
    chunk.split(|c| *c == b'\n').for_each(|line| {
        if let Some((city, measurement_str)) = line.split_once(|&c| c == b';') {
            let measurement = fast_float::parse(measurement_str).unwrap();
            if let Some(existing) = m.get_mut(city) {
                existing.add_mut(measurement);
            } else {
                m.insert(city.to_owned(), MeasurementStats::new(measurement));
            }
        }
    });
    m
}

fn merge_map(
    left: &mut FxHashMap<Vec<u8>, MeasurementStats>,
    right: FxHashMap<Vec<u8>, MeasurementStats>,
) {
    right.iter().for_each(|(k, v)| match left.get_mut(k) {
        Some(measurements) => measurements.merge_mut(v),
        None => {
            left.insert(k.to_vec(), *v);
        }
    });
}

fn handle_fhash_ffloat_read_until_mut_par(file: &str) -> String {
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
                measurement.min,
                measurement.sum / measurement.count as f32,
                measurement.max
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
    fn test_fhash_lookup_read_until_mut_par() {
        let expected = include_str!("../../_data/measurements_1m_result.txt").to_string();
        let actual = handle_fhash_ffloat_read_until_mut_par("../_data/measurements_1m.txt");
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
    fn bench_06_fhash_ffloat_read_until_mut_par(b: &mut Bencher) {
        b.iter(|| handle_fhash_ffloat_read_until_mut_par("../_data/measurements_1m.txt"));
    }
}
