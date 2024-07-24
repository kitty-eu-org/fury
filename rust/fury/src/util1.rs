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

#[cfg(target_feature = "neon")]
use std::arch::aarch64::*;

#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    all(target_arch = "aarch64", target_feature = "neon")
))]
pub(crate) const MIN_DIM_SIZE_SIMD: usize = 8;

// Swapping the high 8 bits and the low 8 bits of a 16-bit value
fn swap_endian(value: u16) -> u16 {
    (value << 8) | (value >> 8)
}

fn utf16_to_utf8(code_point: u16, offset: usize, ptr: *mut u8) -> usize {
    if code_point < 0x80u16 {
        unsafe { ptr.add(1).write(code_point as u8) };
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

#[cfg(target_feature = "neon")]
unsafe fn to_utf8_neon(utf16: &[u16], is_little_endian: bool) -> Result<Vec<u8>, String> {
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

    let remaining = utf16_len % MIN_DIM_SIZE_SIMD;
    let range_end = utf16_len - remaining;
    for i in (0..range_end).step_by(MIN_DIM_SIZE_SIMD) {
        let mut chunk = vld1q_u16(utf16.as_ptr().add(i));
        if !is_little_endian {
            chunk = vorrq_u16(vshlq_n_u16(chunk, 8), vshrq_n_u16(chunk, 8)); // Swap bytes for big-endian
        }
        let masked1 = vandq_u16(chunk, limit1);
        let masked2 = vandq_u16(chunk, limit2);
        let high_surrogate_masked = vandq_u16(
            vcgtq_u16(chunk, surrogate_high_start),
            vcltq_u16(chunk, surrogate_high_end),
        );
        let low_surrogate_masked = vandq_u16(
            vcgtq_u16(chunk, surrogate_low_start),
            vcltq_u16(chunk, surrogate_low_end),
        );
        if vmaxvq_u16(masked1) == 0 {
            for j in 0..MIN_DIM_SIZE_SIMD {
                unsafe {
                    ptr.add(offset).write(utf16[i + j] as u8);
                }
                offset += 1;
            }
            todo!();
        } else if vmaxvq_u16(masked2) == 0 {
            for j in 0..MIN_DIM_SIZE_SIMD {
                offset += utf16_to_utf8(utf16[i + j], offset, ptr)
            }
        } else {
            for j in 0..MIN_DIM_SIZE_SIMD {
                if vmaxvq_u16(high_surrogate_masked) == 0
                    && j + 1 < 8
                    && vmaxvq_u16(low_surrogate_masked) != 0
                {
                    todo!();
                } else {
                    offset += utf16_to_utf8(utf16[i + j], offset, ptr);
                }
            }
        }
    }
    Ok(Vec::new())
}

pub fn to_utf8(utf16: &[u16], is_little_endian: bool) -> Result<Vec<u8>, String> {
    // Pre-allocating capacity to avoid dynamic resizing
    // Longest case: 1 u16 to 3 u8
    let mut utf8_bytes: Vec<u8> = Vec::with_capacity(utf16.len() * 3);
    // For unsafe write to Vec
    let ptr = utf8_bytes.as_mut_ptr();
    let mut offset = 0;
    let mut iter = utf16.iter();
    while let Some(&wc) = iter.next() {
        // Using big endian in this conversion
        let wc = if is_little_endian {
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
                    let wc2 = if is_little_endian {
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
