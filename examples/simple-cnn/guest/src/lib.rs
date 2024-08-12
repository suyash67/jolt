#![cfg_attr(feature = "guest", no_std)]
#![no_main]

extern crate alloc;
use alloc::vec::Vec;

#[jolt::provable]
fn simple_cnn(signal: &[u8], fc_weights_a: &[u8], fc_weights_b: &[u8]) {
    let signal_len = signal.len();
    let fc_weights_a_len = fc_weights_a.len();
    let fc_weights_b_len = fc_weights_b.len();

    // Define the first fully-connected layer
    assert_eq!(fc_weights_a_len % signal_len, 0);
    let fc_weights_a_num_rows = fc_weights_a_len / signal_len;

    let mut conv_output = Vec::<u32>::new();
    for i in 0..fc_weights_a_num_rows {
        let mut row_sum = 0u32;
        for j in 0..signal_len {
            row_sum += (fc_weights_a[i * signal_len + j] as u32) * (signal[j] as u32);
        }
        conv_output.push(row_sum);
    }

    // Define the second fully-connected layer
    assert_eq!(fc_weights_b_len % conv_output.len(), 0);
    let fc_weights_b_num_rows = fc_weights_b_len / conv_output.len();

    let mut output = Vec::<u32>::new();
    for i in 0..fc_weights_b_num_rows {
        let mut row_sum = 0u32;
        for j in 0..conv_output.len() {
            row_sum += (fc_weights_b[i * conv_output.len() + j] as u32) * (conv_output[j] as u32);
        }
        output.push(row_sum);
    }
}
