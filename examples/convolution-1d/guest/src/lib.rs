#![cfg_attr(feature = "guest", no_std)]
#![no_main]

extern crate alloc;
use alloc::vec::Vec;

#[jolt::provable]
fn convolve(signal: &[u8], kernel: &[u8]) {
    let signal_len = signal.len();
    let kernel_len = kernel.len();
    let conv_len = signal_len + kernel_len - 1;

    let mut result = Vec::<u32>::new();
    for _ in 0..conv_len {
        result.push(0);
    }

    for i in 0..signal_len {
        for j in 0..kernel_len {
            result[i + j] += (signal[i] as u32) * (kernel[j] as u32);
        }
    }
}
