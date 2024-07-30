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
pub(crate) const MIN_TO_UTF8_DIM_SIMD: usize = 8;

#[cfg(target_arch = "x86_64")]
pub(crate) const MIN_TO_UTF8_DIM_AVX: usize = 16;

#[cfg(target_feature = "sse2")]
use std::arch::x86_64::*;

use lazy_static::lazy_static;

const PACK_1_2_3_UTF8_BYTES: [[u8; 17]; 256] = [
    [
        12, 2, 3, 1, 6, 7, 5, 10, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        9, 6, 7, 5, 10, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        11, 3, 1, 6, 7, 5, 10, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        10, 0, 6, 7, 5, 10, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        9, 2, 3, 1, 10, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 10, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 3, 1, 10, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 0, 10, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        11, 2, 3, 1, 7, 5, 10, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 7, 5, 10, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        10, 3, 1, 7, 5, 10, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        9, 0, 7, 5, 10, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        10, 2, 3, 1, 4, 10, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 4, 10, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        9, 3, 1, 4, 10, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 0, 4, 10, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        9, 2, 3, 1, 6, 7, 5, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 6, 7, 5, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 3, 1, 6, 7, 5, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 0, 6, 7, 5, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 2, 3, 1, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        3, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 3, 1, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 0, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 2, 3, 1, 7, 5, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 7, 5, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 3, 1, 7, 5, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 0, 7, 5, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 2, 3, 1, 4, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 4, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 3, 1, 4, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 0, 4, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        11, 2, 3, 1, 6, 7, 5, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 6, 7, 5, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        10, 3, 1, 6, 7, 5, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        9, 0, 6, 7, 5, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 2, 3, 1, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 3, 1, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 0, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        10, 2, 3, 1, 7, 5, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 7, 5, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        9, 3, 1, 7, 5, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 0, 7, 5, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        9, 2, 3, 1, 4, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 4, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 3, 1, 4, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 0, 4, 11, 9, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        10, 2, 3, 1, 6, 7, 5, 8, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 6, 7, 5, 8, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        9, 3, 1, 6, 7, 5, 8, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 0, 6, 7, 5, 8, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 2, 3, 1, 8, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 8, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 3, 1, 8, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 0, 8, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        9, 2, 3, 1, 7, 5, 8, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 7, 5, 8, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 3, 1, 7, 5, 8, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 0, 7, 5, 8, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 2, 3, 1, 4, 8, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 4, 8, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 3, 1, 4, 8, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 0, 4, 8, 14, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        9, 2, 3, 1, 6, 7, 5, 10, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 6, 7, 5, 10, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 3, 1, 6, 7, 5, 10, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 0, 6, 7, 5, 10, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 2, 3, 1, 10, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        3, 10, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 3, 1, 10, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 0, 10, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 2, 3, 1, 7, 5, 10, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 7, 5, 10, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 3, 1, 7, 5, 10, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 0, 7, 5, 10, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 2, 3, 1, 4, 10, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 4, 10, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 3, 1, 4, 10, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 0, 4, 10, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 2, 3, 1, 6, 7, 5, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        3, 6, 7, 5, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 3, 1, 6, 7, 5, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 0, 6, 7, 5, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        3, 2, 3, 1, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80,
    ],
    [
        2, 3, 1, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        1, 0, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80,
    ],
    [
        5, 2, 3, 1, 7, 5, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        2, 7, 5, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 3, 1, 7, 5, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        3, 0, 7, 5, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 2, 3, 1, 4, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        1, 4, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80,
    ],
    [
        3, 3, 1, 4, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        2, 0, 4, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 2, 3, 1, 6, 7, 5, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 6, 7, 5, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 3, 1, 6, 7, 5, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 0, 6, 7, 5, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 2, 3, 1, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        2, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80,
    ],
    [
        4, 3, 1, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        3, 0, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 2, 3, 1, 7, 5, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 7, 5, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 3, 1, 7, 5, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 0, 7, 5, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 2, 3, 1, 4, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        3, 4, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 3, 1, 4, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 0, 4, 11, 9, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 2, 3, 1, 6, 7, 5, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 6, 7, 5, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 3, 1, 6, 7, 5, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 0, 6, 7, 5, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 2, 3, 1, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        1, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80,
    ],
    [
        3, 3, 1, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        2, 0, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 2, 3, 1, 7, 5, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        3, 7, 5, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 3, 1, 7, 5, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 0, 7, 5, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 2, 3, 1, 4, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        2, 4, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 3, 1, 4, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        3, 0, 4, 8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        11, 2, 3, 1, 6, 7, 5, 10, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 6, 7, 5, 10, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        10, 3, 1, 6, 7, 5, 10, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        9, 0, 6, 7, 5, 10, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 2, 3, 1, 10, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 10, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 3, 1, 10, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 0, 10, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        10, 2, 3, 1, 7, 5, 10, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 7, 5, 10, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        9, 3, 1, 7, 5, 10, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 0, 7, 5, 10, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        9, 2, 3, 1, 4, 10, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 4, 10, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 3, 1, 4, 10, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 0, 4, 10, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 2, 3, 1, 6, 7, 5, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 6, 7, 5, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 3, 1, 6, 7, 5, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 0, 6, 7, 5, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 2, 3, 1, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        2, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80,
    ],
    [
        4, 3, 1, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        3, 0, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 2, 3, 1, 7, 5, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 7, 5, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 3, 1, 7, 5, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 0, 7, 5, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 2, 3, 1, 4, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        3, 4, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 3, 1, 4, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 0, 4, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        10, 2, 3, 1, 6, 7, 5, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 6, 7, 5, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        9, 3, 1, 6, 7, 5, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 0, 6, 7, 5, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 2, 3, 1, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 3, 1, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 0, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        9, 2, 3, 1, 7, 5, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 7, 5, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 3, 1, 7, 5, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 0, 7, 5, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 2, 3, 1, 4, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 4, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 3, 1, 4, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 0, 4, 11, 9, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        9, 2, 3, 1, 6, 7, 5, 8, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 6, 7, 5, 8, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 3, 1, 6, 7, 5, 8, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 0, 6, 7, 5, 8, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 2, 3, 1, 8, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        3, 8, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 3, 1, 8, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 0, 8, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 2, 3, 1, 7, 5, 8, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 7, 5, 8, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 3, 1, 7, 5, 8, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 0, 7, 5, 8, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 2, 3, 1, 4, 8, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 4, 8, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 3, 1, 4, 8, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 0, 4, 8, 15, 13, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        10, 2, 3, 1, 6, 7, 5, 10, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 6, 7, 5, 10, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        9, 3, 1, 6, 7, 5, 10, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 0, 6, 7, 5, 10, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 2, 3, 1, 10, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 10, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 3, 1, 10, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 0, 10, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        9, 2, 3, 1, 7, 5, 10, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 7, 5, 10, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 3, 1, 7, 5, 10, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 0, 7, 5, 10, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 2, 3, 1, 4, 10, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 4, 10, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 3, 1, 4, 10, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 0, 4, 10, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 2, 3, 1, 6, 7, 5, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 6, 7, 5, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 3, 1, 6, 7, 5, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 0, 6, 7, 5, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 2, 3, 1, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        1, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80,
    ],
    [
        3, 3, 1, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        2, 0, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80,
    ],
    [
        6, 2, 3, 1, 7, 5, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        3, 7, 5, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 3, 1, 7, 5, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 0, 7, 5, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 2, 3, 1, 4, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        2, 4, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80,
    ],
    [
        4, 3, 1, 4, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        3, 0, 4, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        9, 2, 3, 1, 6, 7, 5, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 6, 7, 5, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 3, 1, 6, 7, 5, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 0, 6, 7, 5, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 2, 3, 1, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        3, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 3, 1, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 0, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 2, 3, 1, 7, 5, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 7, 5, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 3, 1, 7, 5, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 0, 7, 5, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 2, 3, 1, 4, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 4, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 3, 1, 4, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 0, 4, 11, 9, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        8, 2, 3, 1, 6, 7, 5, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 6, 7, 5, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 3, 1, 6, 7, 5, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 0, 6, 7, 5, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 2, 3, 1, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        2, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80,
    ],
    [
        4, 3, 1, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        3, 0, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        7, 2, 3, 1, 7, 5, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 7, 5, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 3, 1, 7, 5, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 0, 7, 5, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        6, 2, 3, 1, 4, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        3, 4, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        5, 3, 1, 4, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
    [
        4, 0, 4, 8, 12, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    ],
];

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
    let dup_even = _mm_setr_epi16(
        0x0000, 0x0202, 0x0404, 0x0606, 0x0808, 0x0a0a, 0x0c0c, 0x0e0e,
    );

    let one_mask = _mm_setr_epi16(
        0x0001, 0x0004, 0x0010, 0x0040, 0x0100, 0x0400, 0x1000, 0x4000,
    );

    let two_mask = _mm_setr_epi16(
        0x0002,
        0x0008,
        0x0020,
        0x0080,
        0x0200,
        0x0800,
        0x2000,
        0x8000u16 as i16,
    );

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
            _mm_storeu_si128(
                ptr.add(offset) as *mut __m128i,
                _mm_packus_epi16(*&chunk, *&chunk),
            );
            offset += MIN_TO_UTF8_DIM_SIMD;
        } else if _mm_test_all_zeros(masked2, masked2) == 1 {
            let high_bits = _mm_srli_epi16(chunk, 6);

            let low_bits_mask = _mm_set1_epi16(0b00111111);
            let low_bits_packed_mask = _mm_set1_epi16(0b10000000);

            let low_bits = _mm_and_si128(chunk, low_bits_mask);

            let high_bits_packed = _mm_or_si128(high_bits, low_bits_packed_mask);
            let low_bits_packed = _mm_or_si128(low_bits, low_bits_packed_mask);

            let high_bits_packed_u8 = _mm_packus_epi16(high_bits_packed, high_bits_packed);
            let low_bits_packed_u8 = _mm_packus_epi16(low_bits_packed, low_bits_packed);

            let temp_utf8_buf = _mm_unpacklo_epi8(high_bits_packed_u8, low_bits_packed_u8);

            _mm_storeu_si128(ptr.add(offset) as *mut __m128i, temp_utf8_buf);
            offset += MIN_TO_UTF8_DIM_SIMD * 2;
        } else {
            let t0 = _mm_shuffle_epi8(chunk, dup_even);
            let t1 = _mm_and_si128(t0, _mm_set1_epi16(0b0011111101111111));
            let t2 = _mm_or_si128(t1, _mm_set1_epi16(0b1000000000000000u16 as i16));

            let s0 = _mm_srli_epi16(chunk, 12);
            let s1 = _mm_and_si128(chunk, _mm_set1_epi16(0b0000111111000000));
            let s1s = _mm_slli_epi16(s1, 2);
            let s2 = _mm_or_si128(s0, s1s);
            let s3 = _mm_or_si128(s2, _mm_set1_epi16(0b1100000011100000u16 as i16));
            let v_00800f = _mm_set1_epi16(0x0800);
            let one_or_two_bytes_bytemask = _mm_cmplt_epi16(chunk, v_00800f);
            let m0 = _mm_andnot_si128(
                one_or_two_bytes_bytemask,
                _mm_set1_epi16(0b0100000000000000),
            );
            let s4 = _mm_xor_si128(s3, m0);

            let out0 = _mm_unpacklo_epi16(t2, s4);
            let out1 = _mm_unpackhi_epi16(t2, s4);

            let one_byte_bytemask = _mm_cmplt_epi16(chunk, v_00800f);
            let combined = _mm_or_si128(
                _mm_and_si128(one_byte_bytemask, one_mask),
                _mm_and_si128(one_or_two_bytes_bytemask, two_mask),
            );

            // There's no direct SSE2 equivalent for vaddvq_u16, so we need to sum manually
            let sum_temp = _mm_add_epi16(
                _mm_add_epi16(_mm_srli_si128(combined, 8), combined),
                _mm_add_epi16(_mm_srli_si128(combined, 4), _mm_srli_si128(combined, 12)),
            );
            let mask = _mm_extract_epi16(sum_temp, 0) as u16;

            let (mask0, mask1) = (mask as u8, (mask >> 8) as u8);
            let (row0, row1) = (
                PACK_1_2_3_UTF8_BYTES[mask0 as usize].as_ptr(),
                PACK_1_2_3_UTF8_BYTES[mask1 as usize].as_ptr(),
            );
            let (len0, len1) = (*row0, *row1);
            let (shuffle0, shuffle1) = (
                _mm_loadu_si128(row0.add(1) as *const __m128i),
                _mm_loadu_si128(row1.add(1) as *const __m128i),
            );
            let (utf8_0, utf8_1) = (
                _mm_shuffle_epi8(out0, shuffle0),
                _mm_shuffle_epi8(out1, shuffle1),
            );

            _mm_storeu_si128(ptr.add(offset) as *mut __m128i, utf8_0);
            offset += len0 as usize;
            _mm_storeu_si128(ptr.add(offset) as *mut __m128i, utf8_1);
            offset += len1 as usize;
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
    lazy_static! {
        static ref limit1: uint16x8_t = unsafe { vdupq_n_u16(0x80) };
        static ref limit2: uint16x8_t = unsafe { vdupq_n_u16(0x800) };
        static ref surrogate_high_start: uint16x8_t = unsafe { vdupq_n_u16(0xD800) };
        static ref surrogate_low_end: uint16x8_t = unsafe { vdupq_n_u16(0xDFFF) };
        static ref low_bits_mask: uint16x8_t = unsafe { vdupq_n_u16(0b11_1111) };
        static ref low_bits_packed_mask: uint16x8_t = unsafe { vdupq_n_u16(0b1100_0000) };
        static ref DUP_EVEN: uint16x8_t = unsafe {
            vld1q_u16(
                [
                    0x0000, 0x0202, 0x0404, 0x0606, 0x0808, 0x0a0a, 0x0c0c, 0x0e0e,
                ]
                .as_ptr(),
            )
        };
        static ref ONE_MASK: uint16x8_t = unsafe {
            vld1q_u16(
                [
                    0x0001, 0x0004, 0x0010, 0x0040, 0x0100, 0x0400, 0x1000, 0x4000,
                ]
                .as_ptr(),
            )
        };
        static ref TWO_MASK: uint16x8_t = unsafe {
            vld1q_u16(
                [
                    0x0002, 0x0008, 0x0020, 0x0080, 0x0200, 0x0800, 0x2000, 0x8000,
                ]
                .as_ptr(),
            )
        };
    }

    let remaining = utf16_len % MIN_TO_UTF8_DIM_SIMD;
    let range_end = utf16_len - remaining;
    for i in (0..range_end).step_by(MIN_TO_UTF8_DIM_SIMD) {
        let mut chunk = vld1q_u16(utf16.as_ptr().add(i));
        if !is_little_endian {
            // chunk = vorrq_u16(vshlq_n_u16(chunk, 8), vshrq_n_u16(chunk, 8)); // Swap bytes for big-endian
            chunk = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(chunk)));
        }
        let masked1 = vcgtq_u16(chunk, *limit1);

        let masked2 = vandq_u16(masked1, vcgtq_u16(chunk, *limit2));

        if vmaxvq_u16(masked1) == 0 {
            vst1_u8(ptr.add(offset), vmovn_u16(*&chunk));
            offset += MIN_TO_UTF8_DIM_SIMD;
        } else if vmaxvq_u16(masked2) == 0 {
            // Shift right by 6 bits to get the high 5 bits of each UTF-16 value
            let high_bits = vshrq_n_u16(chunk, 6);

            // Mask out the lower 6 bits of each UTF-16 value
            let low_bits = vandq_u16(chunk, *low_bits_mask);

            // OR with 0b1100_0000 to set the high bits for the first byte of UTF-8
            let high_bits_packed = vorrq_u16(high_bits, *low_bits_packed_mask);

            // OR with 0b1000_0000 to set the high bits for the second byte of UTF-8
            let low_bits_packed = vorrq_u16(low_bits, *low_bits_packed_mask);

            // Interleave the high and low bits to form the UTF-8 bytes
            let temp_utf8_buf = vzip1q_u8(
                vreinterpretq_u8_u16(high_bits_packed),
                vreinterpretq_u8_u16(low_bits_packed),
            );

            // Store the result in the output buffer
            vst1q_u8(ptr, temp_utf8_buf);
            offset += MIN_TO_UTF8_DIM_SIMD * 2;
        } else {
            let t0 = vreinterpretq_u16_u8(vqtbl1q_u8(
                vreinterpretq_u8_u16(*&chunk),
                vreinterpretq_u8_u16(*DUP_EVEN),
            ));
            let t1 = vandq_u16(t0, vmovq_n_u16(0b0011111101111111));
            let t2 = vorrq_u16(t1, vmovq_n_u16(0b1000000000000000));

            let s0 = vshrq_n_u16(*&chunk, 12);
            let s1 = vandq_u16(*&chunk, vmovq_n_u16(0b0000111111000000));
            let s1s = vshlq_n_u16(s1, 2);
            let s2 = vorrq_u16(s0, s1s);
            let s3 = vorrq_u16(s2, vmovq_n_u16(0b1100000011100000));
            let v_07ff = vmovq_n_u16(0x07FF_u16);
            let one_or_two_bytes_bytemask = vcleq_u16(*&chunk, v_07ff);
            let m0 = vbicq_u16(vmovq_n_u16(0b0100000000000000), one_or_two_bytes_bytemask);
            let s4 = veorq_u16(s3, m0);

            let out0 = vreinterpretq_u8_u16(vzip1q_u16(t2, s4));
            let out1 = vreinterpretq_u8_u16(vzip2q_u16(t2, s4));

            let v_007f = vmovq_n_u16(0x007F);
            let one_byte_bytemask = vcleq_u16(*&chunk, v_007f);
            let combined = vorrq_u16(
                vandq_u16(one_byte_bytemask, *ONE_MASK),
                vandq_u16(one_or_two_bytes_bytemask, *TWO_MASK),
            );
            let mask = vaddvq_u16(combined);

            let (mask0, mask1) = (mask as u8, (mask >> 8) as u8);
            let (row0, row1) = (
                PACK_1_2_3_UTF8_BYTES[mask0 as usize].as_ptr(),
                PACK_1_2_3_UTF8_BYTES[mask1 as usize].as_ptr(),
            );
            let (len0, len1) = (*row0, *row1);
            let (shuffle0, shuffle1) = (vld1q_u8(row0.add(1)), vld1q_u8(row1.add(1)));
            let (utf8_0, utf8_1) = (vqtbl1q_u8(out0, shuffle0), vqtbl1q_u8(out1, shuffle1));

            vst1q_u8(ptr.add(offset), utf8_0);
            offset += len0 as usize;
            vst1q_u8(ptr.add(offset), utf8_1);
            offset += len1 as usize;
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

#[cfg(test)]
mod tests {
    use core::str;

    use super::*;

    #[test]
    fn test_to_utf8() {
        let basic_str: Vec<u16> = "Hello, 世界!".encode_utf16().collect();
        let empty_str = Vec::<u16>::new();
        let expected_empty_str = "".as_bytes().to_vec();
        let emoji_str = vec![0xD83Du16, 0xDE00u16]; // 😀 emoji
        let expected_emoji_str = b"\xF0\x9F\x98\x80".to_vec();
        let boundary_str = vec![0x0000u16, 0xFFFFu16];
        let expected_boundary_str = b"\x00\xEF\xBF\xBF".to_vec();
        let new_line_str: Vec<u16> = " \n\t".encode_utf16().collect();
        let expected_new_line_str: Vec<u8> = b" \n\t".to_vec();
        let small_end_str = vec![0x61u16, 0x62u16]; // "ab"
        let expected_small_end_str = b"ab".to_vec();
        let big_end_str = vec![0xFFFEu16, 0xFFFEu16];
        let expected_big_end_str = b"\xEF\xBF\xBE\xEF\xBF\xBE".to_vec();
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
                assert_eq!(
                    str::from_utf8(&(unsafe { to_utf8_avx(&basic_str, true) }).unwrap()),
                    Ok("Hello, 世界!")
                );
                assert_eq!(
                    unsafe { to_utf8_avx(&empty_str, true) },
                    Ok(expected_empty_str.clone())
                );
                assert_eq!(
                    unsafe { to_utf8_avx(&emoji_str, true) },
                    Ok(expected_emoji_str.clone())
                );
                assert_eq!(
                    unsafe { to_utf8_avx(&boundary_str, true) },
                    Ok(expected_boundary_str.clone())
                );
                assert_eq!(
                    unsafe { to_utf8_avx(&new_line_str, true) },
                    Ok(expected_new_line_str.clone())
                );
                assert_eq!(
                    unsafe { to_utf8_avx(&small_end_str, true) },
                    Ok(expected_small_end_str.clone())
                );
                assert_eq!(
                    unsafe { to_utf8_avx(&big_end_str, true) },
                    Ok(expected_big_end_str.clone())
                );
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("sse") {
                assert_eq!(
                    str::from_utf8(&(unsafe { to_utf8_avx(&basic_str, true) }).unwrap()),
                    Ok("Hello, 世界!")
                );
                assert_eq!(
                    unsafe { to_utf8_avx(&empty_str, true) },
                    Ok(expected_empty_str.clone())
                );
                assert_eq!(
                    unsafe { to_utf8_avx(&emoji_str, true) },
                    Ok(expected_emoji_str.clone())
                );
                assert_eq!(
                    unsafe { to_utf8_avx(&boundary_str, true) },
                    Ok(expected_boundary_str.clone())
                );
                assert_eq!(
                    unsafe { to_utf8_avx(&new_line_str, true) },
                    Ok(expected_new_line_str.clone())
                );
                assert_eq!(
                    unsafe { to_utf8_avx(&small_end_str, true) },
                    Ok(expected_small_end_str.clone())
                );
                assert_eq!(
                    unsafe { to_utf8_avx(&big_end_str, true) },
                    Ok(expected_big_end_str.clone())
                );
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                assert_eq!(
                    str::from_utf8(&(unsafe { to_utf8_neon(&basic_str, true) }).unwrap()),
                    Ok("Hello, 世界!")
                );
                assert_eq!(
                    unsafe { to_utf8_neon(&empty_str, true) },
                    Ok(expected_empty_str.clone())
                );
                assert_eq!(
                    unsafe { to_utf8_neon(&emoji_str, true) },
                    Ok(expected_emoji_str.clone())
                );
                assert_eq!(
                    unsafe { to_utf8_neon(&boundary_str, true) },
                    Ok(expected_boundary_str.clone())
                );
                assert_eq!(
                    unsafe { to_utf8_neon(&new_line_str, true) },
                    Ok(expected_new_line_str.clone())
                );
                assert_eq!(
                    unsafe { to_utf8_neon(&small_end_str, true) },
                    Ok(expected_small_end_str.clone())
                );
                assert_eq!(
                    unsafe { to_utf8_neon(&big_end_str, true) },
                    Ok(expected_big_end_str.clone())
                );
            }
        }

        assert_eq!(
            str::from_utf8(&(to_utf8_std(&basic_str, true).unwrap())),
            Ok("Hello, 世界!")
        );
        assert_eq!(to_utf8_std(&empty_str, true), Ok(expected_empty_str));
        assert_eq!(to_utf8_std(&emoji_str, true), Ok(expected_emoji_str));
        assert_eq!(to_utf8_std(&boundary_str, true), Ok(expected_boundary_str));
        assert_eq!(to_utf8_std(&new_line_str, true), Ok(expected_new_line_str));
        assert_eq!(
            to_utf8_std(&small_end_str, true),
            Ok(expected_small_end_str)
        );
        assert_eq!(to_utf8_std(&big_end_str, true), Ok(expected_big_end_str));
    }
}
