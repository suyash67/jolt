pub fn main() {
    let (prove_convolve_and_ip, verify_convolve_and_ip) = guest::build_convolve_and_ip();
    let signal = vec![
        75, 254, 190, 200, 75, 254, 190, 200, 75, 254, 190, 200, 75, 254, 190, 200, 75, 254, 190,
        200, 75, 254, 190, 200, 75, 254, 190, 200, 75, 254, 190, 200,
    ];
    let kernel = vec![4, 6, 8];
    let weights = vec![10; 34];
    let (sum, proof_convolve_and_ip) = prove_convolve_and_ip(&signal, &kernel, &weights);
    let is_valid_convolve = verify_convolve_and_ip(proof_convolve_and_ip);

    println!("valid: {}", is_valid_convolve);
    println!("output: {}", sum);
}
