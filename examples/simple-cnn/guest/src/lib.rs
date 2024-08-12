#![cfg_attr(feature = "guest", no_std)]
#![no_main]

extern crate alloc;
use alloc::vec::Vec;

#[jolt::provable]
fn simple_cnn(signal: &[u8], conv_kernel: &[u8], fc_weights: &[u8]) {
    let signal_len = signal.len();
    let conv_kernel_len = conv_kernel.len();
    let fc_weights_len = fc_weights.len();

    // Define the convolutional layer with a single filter
    assert!(signal_len >= conv_kernel_len);
    let conv_output_len = signal_len - conv_kernel_len + 1;
    let mut conv_output = Vec::<u32>::new();

    for i in 0..conv_output_len {
        let mut sum = 0u32;
        for j in 0..conv_kernel_len {
            sum += (signal[i + j] as u32) * (conv_kernel[j] as u32);
        }
        conv_output.push(sum);
    }

    // Define the fully-connected layer
    assert_eq!(fc_weights_len % conv_output_len, 0);
    let fc_weights_num_rows = fc_weights_len / conv_output_len;

    let mut output = Vec::<u32>::new();
    for i in 0..fc_weights_num_rows {
        let mut row_sum = 0u32;
        for j in 0..conv_output_len {
            row_sum += (fc_weights[i * conv_output_len + j] as u32) * (conv_output[j] as u32);
        }
        output.push(row_sum);
    }
}
