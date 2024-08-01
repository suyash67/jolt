#![cfg_attr(feature = "guest", no_std)]
#![no_main]

#[jolt::provable]
fn inner_prod(a_vec: &[u8], b_vec: &[u8]) -> u32 {
    // Compute the inner-product <a, b> using normal Rust code.
    let mut sum = 0u32;
    let a_len = a_vec.len();
    for i in 0..a_len {
        let c: u32 = (a_vec[i as usize] as u32) * (b_vec[i as usize] as u32);
        sum += c;
    }
    sum
}
