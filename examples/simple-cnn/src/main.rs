pub fn main() {
    let (prove_simple_cnn, verify_simple_cnn) = guest::build_simple_cnn();
    let signal = vec![250; 256];
    let kernel = vec![15, 25, 35];
    let weights_fc = vec![31; (256 - 3 + 1) * 8];
    let (_, proof_simple_cnn) = prove_simple_cnn(&signal, &kernel, &weights_fc);
    let is_valid_convolve = verify_simple_cnn(proof_simple_cnn);

    println!("valid: {}", is_valid_convolve);
}
