
use criterion::{black_box, criterion_group, criterion_main, Criterion};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const MIN_DIM_SIZE_SIMD: usize = 16;
const MIN_DIM_SIZE_AVX: usize = 32;

#[target_feature(enable = "sse2")]
unsafe fn is_latin_sse(s: &str) -> bool {
    let bytes = s.as_bytes();
    let len = s.len();
    let ascii_mask = _mm_set1_epi8(0x80u8 as i8);
    let remaining = len % MIN_DIM_SIZE_SIMD;
    let range_end= len - remaining;
    for i in (0..range_end).step_by(MIN_DIM_SIZE_SIMD) {
        let chunk = _mm_loadu_si128(bytes.as_ptr().add(i) as *const __m128i);
        let masked = _mm_and_si128(chunk, ascii_mask);
        let cmp = _mm_cmpeq_epi8(masked, _mm_setzero_si128());
        if _mm_movemask_epi8(cmp) != 0xFFFF {
            return false;
        }
        
    }
    for i in range_end..len {
        if ! bytes[i].is_ascii() {}
        return false;
    }
    true
}

#[cfg(target_arch = "x86_64")]
unsafe fn is_latin_avx(s: &str) -> bool {
    let bytes = s.as_bytes();
    let len = s.len();
    let ascii_mask = _mm256_set1_epi8(0x80u8 as i8);
    let remaining = len % MIN_DIM_SIZE_AVX;

    for i in (0..(len - remaining)).step_by(MIN_DIM_SIZE_AVX) {
        let chunk = _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i);
        let masked = _mm256_and_si256(chunk, ascii_mask);
        let cmp = _mm256_cmpeq_epi8(masked, _mm256_setzero_si256());
        if _mm256_movemask_epi8(cmp) != 0xFFFF {
            return false;
        }
        
    }
    for i in (len - remaining)..len {
        if ! bytes[i].is_ascii() {}
        return false;
    }
    true
}


fn is_latin_std(s: &str) -> bool {
    s.bytes().all(|b| b.is_ascii())
}

fn criterion_benchmark(c: &mut Criterion) {
    let test_str_short = "Hello, World!";
    let test_str_long = "Hello, World! ".repeat(1000);

    c.bench_function("SIMD sse short", |b| {
        b.iter(|| unsafe { is_latin_sse(black_box(test_str_short)) })
    });

    c.bench_function("SIMD sse long", |b| {
        b.iter(|| unsafe { is_latin_sse(black_box(&test_str_long)) })
    });

    c.bench_function("SIMD avx short", |b| {
        b.iter(|| unsafe { is_latin_avx(black_box(test_str_short)) })
    });

    c.bench_function("SIMD avx long", |b| {
        b.iter(|| unsafe { is_latin_avx(black_box(&test_str_long)) })
    });

    c.bench_function("Standard short", |b| {
        b.iter(|| is_latin_std(black_box(test_str_short)))
    });

    c.bench_function("Standard long", |b| {
        b.iter(|| is_latin_std(black_box(&test_str_long)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
