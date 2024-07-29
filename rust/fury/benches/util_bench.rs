// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::ptr;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;

#[cfg(target_feature = "neon")]
use std::arch::aarch64::*;

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "aarch64", target_feature = "neon")
))]
pub(crate) const MIN_TO_UTF8_DIM_SIMD: usize = 8;

#[cfg(target_arch = "x86_64")]
pub(crate) const MIN_TO_UTF8_DIM_AVX: usize = 16;

#[cfg(target_feature = "sse2")]
use std::arch::x86_64::*;

// Swapping the high 8 bits and the low 8 bits of a 16-bit value
fn swap_endian(value: u16) -> u16 {
    (value << 8) | (value >> 8)
}

fn utf16_to_utf8(code_point: u16, offset: usize, ptr: *mut u8) -> usize {
    if code_point < 0x80u16 {
        unsafe { ptr.add(offset).write(code_point as u8) };
        1
    } else if code_point < 0x800u16 {
        let bytes = [
            (code_point >> 6 & 0b1_1111) as u8 | 0b1100_0000,
            (code_point & 0b11_1111) as u8 | 0b1000_0000,
        ];
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(offset), 2);
        }
        2
    } else {
        let bytes = [
            (code_point >> 12 | 0b1110_0000) as u8,
            (code_point >> 6 & 0b11_1111) as u8 | 0b1000_0000,
            (code_point & 0b11_1111) as u8 | 0b1000_0000,
        ];
        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(offset), 3);
        }
        3
    }
}

fn utf16_surrogate_pair_to_utf8(high: u16, low: u16, offset: usize, ptr: *mut u8) -> usize {
    // utf16 to unicode
    let code_point = ((((high as u32) - 0xd800) << 10) | ((low as u32) - 0xdc00)) + 0x10000;
    // 11110??? 10?????? 10?????? 10??????
    // Need 21 bit suffix of code_point
    let bytes = [
        (code_point >> 18 & 0b111) as u8 | 0b1111_0000,
        (code_point >> 12 & 0b11_1111) as u8 | 0b1000_0000,
        (code_point >> 6 & 0b11_1111) as u8 | 0b1000_0000,
        (code_point & 0b11_1111) as u8 | 0b1000_0000,
    ];
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(offset), 4);
    }
    4
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn to_utf8_avx(utf16: &[u16], is_little_endian: bool) -> Result<Vec<u8>, String> {
    let utf16_len = utf16.len();
    let mut offset = 0;
    let mut utf8_bytes: Vec<u8> = Vec::with_capacity(utf16_len * 3);
    let ptr = utf8_bytes.as_mut_ptr();
    let limit1 = _mm256_set1_epi16(0x80);
    let limit2 = _mm256_set1_epi16(0x800);
    let surrogate_high_start = _mm256_set1_epi16(0xD800u16 as i16);
    let surrogate_high_end = _mm256_set1_epi16(0xDBFFu16 as i16);
    let surrogate_low_start = _mm256_set1_epi16(0xDC00u16 as i16);
    let surrogate_low_end = _mm256_set1_epi16(0xDFFFu16 as i16);

    let remaining = utf16_len % MIN_TO_UTF8_DIM_AVX;
    let range_end = utf16_len - remaining;
    for i in (0..range_end).step_by(MIN_TO_UTF8_DIM_AVX) {
        let mut chunk = _mm256_loadu_si256(utf16.as_ptr().add(i) as *const __m256i);
        if !is_little_endian {
            chunk = _mm256_or_si256(_mm256_slli_epi16(chunk, 8), _mm256_srli_epi16(chunk, 8));
            // Swap bytes for big-endian
        }
        let masked1 = _mm256_cmpgt_epi16(chunk, limit1);
        let masked2 = _mm256_cmpgt_epi16(chunk, limit2);
        let high_surrogate_masked = _mm256_and_si256(
            _mm256_cmpgt_epi16(chunk, surrogate_high_start),
            _mm256_cmpgt_epi16(chunk, surrogate_high_end),
        );
        let low_surrogate_masked = _mm256_and_si256(
            _mm256_cmpgt_epi16(chunk, surrogate_low_start),
            _mm256_cmpgt_epi16(chunk, surrogate_low_end),
        );
        if _mm256_testz_si256(masked1, masked1) == 1 {
            for j in 0..MIN_TO_UTF8_DIM_AVX {
                unsafe {
                    ptr.add(offset).write(utf16[i + j] as u8);
                }
                offset += 1;
            }
        } else if _mm256_testz_si256(masked2, masked2) == 1 {
            for j in 0..MIN_TO_UTF8_DIM_AVX {
                offset += utf16_to_utf8(utf16[i + j], offset, ptr)
            }
        } else {
            for j in 0..MIN_TO_UTF8_DIM_AVX {
                if _mm256_testz_si256(high_surrogate_masked, high_surrogate_masked) == 1
                    && j + 1 < MIN_TO_UTF8_DIM_AVX
                    && _mm256_testz_si256(low_surrogate_masked, low_surrogate_masked) != 1
                {
                    if !(0xdc00..=0xdfff).contains(&utf16[i + j]) {
                        return Err("Invalid UTF-16 string: wrong surrogate pair".to_string());
                    }
                    offset +=
                        utf16_surrogate_pair_to_utf8(utf16[i + j], utf16[i + j + 1], offset, ptr);
                } else {
                    offset += utf16_to_utf8(utf16[i + j], offset, ptr);
                }
            }
        }
    }

    let mut i = range_end;
    while i < utf16_len {
        if i + 1 < utf16_len
            && utf16[i] >= 0xD800
            && utf16[i] <= 0xDBFF
            && utf16[i + 1] >= 0xDC00
            && utf16[i + 1] <= 0xDFFF
        {
            if !(0xDC00..=0xDFFF).contains(&utf16[i + 1]) {
                return Err("Invalid UTF-16 string: wrong surrogate pair".to_string());
            }
            offset += utf16_surrogate_pair_to_utf8(utf16[i], utf16[i + 1], offset, ptr);
            i += 2;
        } else {
            offset += utf16_to_utf8(utf16[i], offset, ptr);
            i += 1;
        }
    }
    unsafe {
        // As ptr.write don't change the length
        utf8_bytes.set_len(offset);
    }
    Ok(utf8_bytes)
}

#[cfg(target_feature = "sse2")]
pub unsafe fn to_utf8_sse(utf16: &[u16], is_little_endian: bool) -> Result<Vec<u8>, String> {
    let utf16_len = utf16.len();
    let mut offset = 0;
    let mut utf8_bytes: Vec<u8> = Vec::with_capacity(utf16_len * 3);
    let ptr = utf8_bytes.as_mut_ptr();
    let limit1 = _mm_set1_epi16(0x80);
    let limit2 = _mm_set1_epi16(0x800);
    let surrogate_high_start = _mm_set1_epi16(0xD800u16 as i16);
    let surrogate_low_end = _mm_set1_epi16(0xDFFFu16 as i16);

    let remaining = utf16_len % MIN_TO_UTF8_DIM_SIMD;
    let range_end = utf16_len - remaining;
    for i in (0..range_end).step_by(MIN_TO_UTF8_DIM_SIMD) {
        let mut chunk = _mm_loadu_si128(utf16.as_ptr().add(i) as *const __m128i);
        if !is_little_endian {
            chunk = _mm_or_si128(_mm_slli_epi16(chunk, 8), _mm_slli_epi16(chunk, 8));
            // Swap bytes for big-endian
        }
        let masked1 = _mm_cmplt_epi16(chunk, limit1);
        let masked2 = _mm_andnot_si128(masked1, _mm_cmplt_epi16(chunk, limit2));
        let is_surrogate = _mm_and_si128(
            _mm_cmpgt_epi16(chunk, surrogate_high_start),
            _mm_cmplt_epi16(chunk, surrogate_low_end),
        );

        if _mm_test_all_zeros(masked1, masked1) == 1 {
            unsafe {
                ptr::copy_nonoverlapping(
                    utf16[i..i + MIN_TO_UTF8_DIM_SIMD].as_ptr() as *const u8,
                    ptr.add(offset),
                    MIN_TO_UTF8_DIM_SIMD,
                );
            }
            offset += MIN_TO_UTF8_DIM_SIMD;
        } else if _mm_test_all_zeros(masked2, masked2) == 1 {
            for j in 0..MIN_TO_UTF8_DIM_SIMD {
                let bytes = [
                    (utf16[i + j] >> 6 & 0b1_1111) as u8 | 0b1100_0000,
                    (utf16[i + j] & 0b11_1111) as u8 | 0b1000_0000,
                ];
                ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    ptr.add(offset),
                    2,
                );
                offset += 2
            }
        } else {
            for j in 0..MIN_TO_UTF8_DIM_SIMD {
                if _mm_test_all_zeros(is_surrogate, is_surrogate) == 1
                    && j + 1 < MIN_TO_UTF8_DIM_SIMD
                    && (0xdc00..=0xdfff).contains(&utf16[i + j + 1])
                {
                    offset +=
                        utf16_surrogate_pair_to_utf8(utf16[i + j], utf16[i + j + 1], offset, ptr);
                } else {
                    offset += utf16_to_utf8(utf16[i + j], offset, ptr);
                }
            }
        }
    }

    let mut i = range_end;
    while i < utf16_len {
        if i + 1 < utf16_len
            && utf16[i] >= 0xD800
            && utf16[i] <= 0xDBFF
            && utf16[i + 1] >= 0xDC00
            && utf16[i + 1] <= 0xDFFF
        {
            if !(0xDC00..=0xDFFF).contains(&utf16[i + 1]) {
                return Err("Invalid UTF-16 string: wrong surrogate pair".to_string());
            }
            offset += utf16_surrogate_pair_to_utf8(utf16[i], utf16[i + 1], offset, ptr);
            i += 2;
        } else {
            offset += utf16_to_utf8(utf16[i], offset, ptr);
            i += 1;
        }
    }
    unsafe {
        // As ptr.write don't change the length
        utf8_bytes.set_len(offset);
    }
    Ok(utf8_bytes)
}

#[cfg(target_feature = "neon")]
pub unsafe fn to_utf8_neon(utf16: &[u16], is_little_endian: bool) -> Result<Vec<u8>, String> {
    let utf16_len = utf16.len();
    let mut offset = 0;
    let mut utf8_bytes: Vec<u8> = Vec::with_capacity(utf16_len * 3);
    let ptr = utf8_bytes.as_mut_ptr();
    let limit1 = vdupq_n_u16(0x80);
    let limit2 = vdupq_n_u16(0x800);
    let surrogate_high_start = vdupq_n_u16(0xD800);
    let surrogate_high_end = vdupq_n_u16(0xDBFF);
    let surrogate_low_start = vdupq_n_u16(0xDC00);
    let surrogate_low_end = vdupq_n_u16(0xDFFF);

    let remaining = utf16_len % MIN_TO_UTF8_DIM_SIMD;
    let range_end = utf16_len - remaining;
    for i in (0..range_end).step_by(MIN_TO_UTF8_DIM_SIMD) {
        let mut chunk = vld1q_u16(utf16.as_ptr().add(i));
        if !is_little_endian {
            chunk = vorrq_u16(vshlq_n_u16(chunk, 8), vshrq_n_u16(chunk, 8)); // Swap bytes for big-endian
        }
        let masked1 = vcgtq_u16(chunk, limit1);
        let masked2 = vcgtq_u16(chunk, limit2);
        let high_surrogate_masked = vandq_u16(
            vcgtq_u16(chunk, surrogate_high_start),
            vcltq_u16(chunk, surrogate_high_end),
        );
        let low_surrogate_masked = vandq_u16(
            vcgtq_u16(chunk, surrogate_low_start),
            vcltq_u16(chunk, surrogate_low_end),
        );
        let res = vmaxvq_u16(masked1);
        if vmaxvq_u16(masked1) == 0 {
            for j in 0..MIN_TO_UTF8_DIM_SIMD {
                unsafe {
                    ptr.add(offset).write(utf16[i + j] as u8);
                }
                offset += 1;
            }
        } else if vmaxvq_u16(masked2) == 0 {
            for j in 0..MIN_TO_UTF8_DIM_SIMD {
                offset += utf16_to_utf8(utf16[i + j], offset, ptr)
            }
        } else {
            for j in 0..MIN_TO_UTF8_DIM_SIMD {
                if vmaxvq_u16(high_surrogate_masked) == 0
                    && j + 1 < MIN_TO_UTF8_DIM_SIMD
                    && vmaxvq_u16(low_surrogate_masked) != 0
                {
                    if !(0xdc00..=0xdfff).contains(&utf16[i + j]) {
                        return Err("Invalid UTF-16 string: wrong surrogate pair".to_string());
                    }
                    offset +=
                        utf16_surrogate_pair_to_utf8(utf16[i + j], utf16[i + j + 1], offset, ptr);
                } else {
                    offset += utf16_to_utf8(utf16[i + j], offset, ptr);
                }
            }
        }
    }
    let mut i = range_end;
    while i < utf16_len {
        if i + 1 < utf16_len
            && utf16[i] >= 0xD800
            && utf16[i] <= 0xDBFF
            && utf16[i + 1] >= 0xDC00
            && utf16[i + 1] <= 0xDFFF
        {
            if !(0xDC00..=0xDFFF).contains(&utf16[i + 1]) {
                return Err("Invalid UTF-16 string: wrong surrogate pair".to_string());
            }
            offset += utf16_surrogate_pair_to_utf8(utf16[i], utf16[i + 1], offset, ptr);
            i += 2;
        } else {
            offset += utf16_to_utf8(utf16[i], offset, ptr);
            i += 1;
        }
    }
    unsafe {
        // As ptr.write don't change the length
        utf8_bytes.set_len(offset);
    }
    Ok(utf8_bytes)
}

pub fn to_utf8_std(utf16: &[u16], is_little_endian: bool) -> Result<Vec<u8>, String> {
    // Pre-allocating capacity to avoid dynamic resizing
    // Longest case: 1 u16 to 3 u8
    let mut utf8_bytes: Vec<u8> = Vec::with_capacity(utf16.len() * 3);
    // For unsafe write to Vec
    let ptr = utf8_bytes.as_mut_ptr();
    let mut offset = 0;
    let mut iter = utf16.iter();
    while let Some(&wc) = iter.next() {
        // Using big endian in this conversion
        let wc = if !is_little_endian {
            swap_endian(wc)
        } else {
            wc
        };
        match wc {
            code_point if code_point < 0x80 => {
                // 1-byte UTF-8
                // [0000|0000|0ccc|cccc] => [0ccc|cccc]
                unsafe {
                    ptr.add(offset).write(code_point as u8);
                }
                offset += 1;
            }
            code_point if code_point < 0x800 => {
                // 2-byte UTF-8
                // [0000|0bbb|bbcc|cccc] => [110|bbbbb], [10|cccccc]
                let bytes = [
                    (code_point >> 6 & 0b1_1111) as u8 | 0b1100_0000,
                    (code_point & 0b11_1111) as u8 | 0b1000_0000,
                ];
                unsafe {
                    ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(offset), 2);
                }
                offset += 2;
            }
            wc1 if (0xd800..=0xdbff).contains(&wc1) => {
                // Surrogate pair (4-byte UTF-8)
                // Need extra u16, 2 u16 -> 4 u8
                if let Some(&wc2) = iter.next() {
                    let wc2 = if !is_little_endian {
                        swap_endian(wc2)
                    } else {
                        wc2
                    };
                    if !(0xdc00..=0xdfff).contains(&wc2) {
                        return Err("Invalid UTF-16 string: wrong surrogate pair".to_string());
                    }
                    // utf16 to unicode
                    let code_point =
                        ((((wc1 as u32) - 0xd800) << 10) | ((wc2 as u32) - 0xdc00)) + 0x10000;
                    // 11110??? 10?????? 10?????? 10??????
                    // Need 21 bit suffix of code_point
                    let bytes = [
                        (code_point >> 18 & 0b111) as u8 | 0b1111_0000,
                        (code_point >> 12 & 0b11_1111) as u8 | 0b1000_0000,
                        (code_point >> 6 & 0b11_1111) as u8 | 0b1000_0000,
                        (code_point & 0b11_1111) as u8 | 0b1000_0000,
                    ];
                    unsafe {
                        ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(offset), 4);
                    }
                    offset += 4;
                } else {
                    return Err("Invalid UTF-16 string: missing surrogate pair".to_string());
                }
            }
            _ => {
                // 3-byte UTF-8, 1 u16 -> 3 u8
                // [aaaa|bbbb|bbcc|cccc] => [1110|aaaa], [10|bbbbbb], [10|cccccc]
                // Need 16 bit suffix of wc, as same as wc itself
                let bytes = [
                    (wc >> 12 | 0b1110_0000) as u8,
                    (wc >> 6 & 0b11_1111) as u8 | 0b1000_0000,
                    (wc & 0b11_1111) as u8 | 0b1000_0000,
                ];
                unsafe {
                    ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.add(offset), 3);
                }
                offset += 3;
            }
        }
    }
    unsafe {
        // As ptr.write don't change the length
        utf8_bytes.set_len(offset);
    }
    Ok(utf8_bytes)
}

pub fn to_utf8(utf16: &[u16], is_little_endian: bool) -> Result<Vec<u8>, String> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx")
            && is_x86_feature_detected!("fma")
            && utf16.len() >= MIN_TO_UTF8_DIM_AVX
        {
            return unsafe { to_utf8_avx(utf16, is_little_endian) };
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("sse") && utf16.len() >= MIN_TO_UTF8_DIM_SIMD {
            return unsafe { to_utf8_sse(utf16, is_little_endian) };
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") && utf16.len() >= MIN_TO_UTF8_DIM_SIMD {
            return unsafe { to_utf8_neon(utf16, is_little_endian) };
        }
    }
    to_utf8_std(utf16, is_little_endian)
}

fn generate_random_utf16_string(length: usize) -> Vec<u16> {
    let mut rng = rand::thread_rng();
    (0..length).map(|_| rng.gen_range(0x0000..0xD7FF)).collect()
}

fn criterion_benchmark(c: &mut Criterion) {
    let test_str_short = generate_random_utf16_string(100).repeat(1000);
    let test_str_long = generate_random_utf16_string(1000).repeat(1000);

    #[cfg(target_feature = "sse2")]
    c.bench_function("SIMD sse short", |b| {
        b.iter(|| unsafe { to_utf8_sse(black_box(&test_str_short), true) })
    });
    #[cfg(target_feature = "sse2")]
    c.bench_function("SIMD sse long", |b| {
        b.iter(|| unsafe { to_utf8_sse(black_box(&test_str_long), true) })
    });

    #[cfg(target_feature = "avx2")]
    c.bench_function("SIMD avx short", |b| {
        b.iter(|| unsafe { to_utf8_avx(black_box(&test_str_short), true) })
    });
    #[cfg(target_feature = "avx2")]
    c.bench_function("SIMD avx long", |b| {
        b.iter(|| unsafe { to_utf8_avx(black_box(&test_str_long), true) })
    });

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    c.bench_function("SIMD neon short", |b| {
        b.iter(|| unsafe { to_utf8_neon(black_box(&test_str_short), true) })
    });
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    c.bench_function("SIMD neon long", |b| {
        b.iter(|| unsafe { to_utf8_neon(black_box(&test_str_long), true) })
    });

    c.bench_function("Standard short", |b| {
        b.iter(|| to_utf8_std(black_box(&test_str_short), true))
    });

    c.bench_function("Standard long", |b| {
        b.iter(|| to_utf8_std(black_box(&test_str_long), true))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
