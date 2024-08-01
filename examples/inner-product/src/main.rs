pub fn main() {
    let (prove_inner_prod, verify_inner_prod) = guest::build_inner_prod();

    let a_vec: &[u8] = &[5u8; 32];
    let b_vec: &[u8] = &[9u8; 32];
    let (output, proof) = prove_inner_prod(a_vec, b_vec);
    let is_valid = verify_inner_prod(proof);

    println!("output: {}", output);
    println!("valid: {}", is_valid);
}
