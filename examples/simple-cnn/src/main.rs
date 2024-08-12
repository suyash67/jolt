pub fn main() {
    let (prove_simple_cnn, verify_simple_cnn) = guest::build_simple_cnn();
    let signal = vec![250; 256];
    let weights_fc_a = vec![30; 64 * 256];
    let weights_fc_b = vec![31; 32 * 64];
    let (_, proof_simple_cnn) = prove_simple_cnn(&signal, &weights_fc_a, &weights_fc_b);
    let is_valid_convolve = verify_simple_cnn(proof_simple_cnn);

    println!("valid: {}", is_valid_convolve);
}
