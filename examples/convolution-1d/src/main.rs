pub fn main() {
    let (prove_convolve, verify_convolve) = guest::build_convolve();
    let signal = vec![
        75, 254, 190, 200, 75, 254, 190, 200, 75, 254, 190, 200, 75, 254, 190, 200, 75, 254, 190,
        200, 75, 254, 190, 200, 75, 254, 190, 200, 75, 254, 190, 200,
    ];
    let kernel = vec![4, 6, 8];
    let (_, proof_convolve) = prove_convolve(&signal, &kernel);
    let is_valid_convolve = verify_convolve(proof_convolve);

    println!("valid: {}", is_valid_convolve);
}
