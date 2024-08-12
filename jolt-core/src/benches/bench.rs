use crate::field::JoltField;
use crate::host::{self, Program};
use crate::jolt::vm::rv32i_vm::{RV32IJoltVM, C, M};
use crate::jolt::vm::Jolt;
use crate::poly::commitment::commitment_scheme::CommitmentScheme;
use crate::poly::commitment::hyperkzg::HyperKZG;
use crate::poly::commitment::hyrax::HyraxScheme;
use crate::poly::commitment::zeromorph::Zeromorph;
use ark_bn254::{Bn254, Fr, G1Projective};
use rand::Rng;
use serde::Serialize;

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum PCSType {
    Hyrax,
    Zeromorph,
    HyperKZG,
}

#[derive(Debug, Copy, Clone, clap::ValueEnum)]
pub enum BenchType {
    Fibonacci,
    Sha2,
    Sha3,
    Sha2Chain,
    InnerProduct,
    Conv1d,
    MixedConvIp,
    SimpleCNN,
}

#[allow(unreachable_patterns)] // good errors on new BenchTypes
pub fn benchmarks(
    pcs_type: PCSType,
    bench_type: BenchType,
    _num_cycles: Option<usize>,
    _memory_size: Option<usize>,
    _bytecode_size: Option<usize>,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)> {
    match pcs_type {
        PCSType::Hyrax => match bench_type {
            BenchType::Sha2 => sha2::<Fr, HyraxScheme<G1Projective>>(),
            BenchType::Sha3 => sha3::<Fr, HyraxScheme<G1Projective>>(),
            BenchType::Sha2Chain => sha2chain::<Fr, HyraxScheme<G1Projective>>(),
            BenchType::Fibonacci => fibonacci::<Fr, HyraxScheme<G1Projective>>(),
            BenchType::InnerProduct => inner_product::<Fr, HyraxScheme<G1Projective>>(),
            BenchType::Conv1d => conv_1d::<Fr, HyraxScheme<G1Projective>>(),
            BenchType::MixedConvIp => mixed_conv_ip::<Fr, HyraxScheme<G1Projective>>(),
            BenchType::SimpleCNN => simple_cnn::<Fr, HyraxScheme<G1Projective>>(),
            _ => panic!("BenchType does not have a mapping"),
        },
        PCSType::Zeromorph => match bench_type {
            BenchType::Sha2 => sha2::<Fr, Zeromorph<Bn254>>(),
            BenchType::Sha3 => sha3::<Fr, Zeromorph<Bn254>>(),
            BenchType::Sha2Chain => sha2chain::<Fr, Zeromorph<Bn254>>(),
            BenchType::Fibonacci => fibonacci::<Fr, Zeromorph<Bn254>>(),
            BenchType::InnerProduct => inner_product::<Fr, Zeromorph<Bn254>>(),
            BenchType::Conv1d => conv_1d::<Fr, Zeromorph<Bn254>>(),
            BenchType::MixedConvIp => mixed_conv_ip::<Fr, Zeromorph<Bn254>>(),
            BenchType::SimpleCNN => simple_cnn::<Fr, Zeromorph<Bn254>>(),
            _ => panic!("BenchType does not have a mapping"),
        },
        PCSType::HyperKZG => match bench_type {
            BenchType::Sha2 => sha2::<Fr, HyperKZG<Bn254>>(),
            BenchType::Sha3 => sha3::<Fr, HyperKZG<Bn254>>(),
            BenchType::Sha2Chain => sha2chain::<Fr, HyperKZG<Bn254>>(),
            BenchType::Fibonacci => fibonacci::<Fr, HyperKZG<Bn254>>(),
            BenchType::InnerProduct => inner_product::<Fr, HyperKZG<Bn254>>(),
            BenchType::Conv1d => conv_1d::<Fr, HyperKZG<Bn254>>(),
            BenchType::MixedConvIp => mixed_conv_ip::<Fr, HyperKZG<Bn254>>(),
            BenchType::SimpleCNN => simple_cnn::<Fr, HyperKZG<Bn254>>(),
            _ => panic!("BenchType does not have a mapping"),
        },
        _ => panic!("PCS Type does not have a mapping"),
    }
}

fn fibonacci<F, PCS>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    prove_example::<u32, PCS, F>("fibonacci-guest", &9u32)
}

fn sha2<F, PCS>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    prove_example::<Vec<u8>, PCS, F>("sha2-guest", &vec![5u8; 2048])
}

fn sha3<F, PCS>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    prove_example::<Vec<u8>, PCS, F>("sha3-guest", &vec![5u8; 2048])
}

#[allow(dead_code)]
fn serialize_and_print_size(name: &str, item: &impl ark_serialize::CanonicalSerialize) {
    use std::fs::File;
    let mut file = File::create("temp_file").unwrap();
    item.serialize_compressed(&mut file).unwrap();
    let file_size_bytes = file.metadata().unwrap().len();
    let file_size_kb = file_size_bytes as f64 / 1024.0;
    let file_size_mb = file_size_kb / 1024.0;
    println!("{:<30} : {:.3} MB", name, file_size_mb);
}

fn generate_proof_and_verify<F, PCS>(
    mut program: Program,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    let mut tasks = Vec::new();

    let task = move || {
        let (bytecode, memory_init) = program.decode();
        let (io_device, trace, circuit_flags) = program.trace();

        let preprocessing: crate::jolt::vm::JoltPreprocessing<F, PCS> =
            RV32IJoltVM::preprocess(bytecode.clone(), memory_init, 1 << 20, 1 << 20, 1 << 22);

        let (jolt_proof, jolt_commitments) = <RV32IJoltVM as Jolt<_, PCS, C, M>>::prove(
            io_device,
            trace,
            circuit_flags,
            preprocessing.clone(),
        );

        println!("Proof sizing:");
        serialize_and_print_size("jolt_commitments", &jolt_commitments);
        serialize_and_print_size("jolt_proof", &jolt_proof);
        serialize_and_print_size(" jolt_proof.r1cs", &jolt_proof.r1cs);
        serialize_and_print_size(" jolt_proof.bytecode", &jolt_proof.bytecode);
        serialize_and_print_size(
            " jolt_proof.read_write_memory",
            &jolt_proof.read_write_memory,
        );
        serialize_and_print_size(
            " jolt_proof.instruction_lookups",
            &jolt_proof.instruction_lookups,
        );

        let verification_result = RV32IJoltVM::verify(preprocessing, jolt_proof, jolt_commitments);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    };

    tasks.push((
        tracing::info_span!("Example_E2E"),
        Box::new(task) as Box<dyn FnOnce()>,
    ));

    tasks
}

fn prove_example<T: Serialize, PCS, F>(
    example_name: &str,
    input: &T,
) -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    let mut program = host::Program::new(example_name);
    program.set_input(input);

    generate_proof_and_verify::<F, PCS>(program)
}

fn inner_product<F, PCS>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    let mut program = host::Program::new("inner-product-guest");
    let input_a = vec![255u8; 32768];
    let input_b = vec![254u8; 32768];
    program.set_input(&input_a);
    program.set_input(&input_b);

    generate_proof_and_verify::<F, PCS>(program)
}

fn conv_1d<F, PCS>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    let mut program = host::Program::new("conv-1d-guest");
    let signal = vec![255u8; 1024];
    let kernel = vec![31u8; 32];
    program.set_input(&signal);
    program.set_input(&kernel);

    generate_proof_and_verify::<F, PCS>(program)
}

fn mixed_conv_ip<F, PCS>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    let mut program = host::Program::new("mixed-conv-ip-guest");
    let signal_len = 1 << 11;
    let kernel_len = 32;
    let signal = vec![255u8; signal_len];
    let kernel = vec![31u8; kernel_len];
    let weights = vec![254u8; signal_len + kernel_len - 1];
    program.set_input(&signal);
    program.set_input(&kernel);
    program.set_input(&weights);

    generate_proof_and_verify::<F, PCS>(program)
}

fn generate_random_bytes(len: usize) -> Vec<u8> {
    let mut rng = rand::thread_rng(); // Create a random number generator
    (0..len).map(|_| rng.gen()).collect() // Generate random numbers and collect them into a vector
}

fn simple_cnn<F, PCS>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    let mut program = host::Program::new("simple-cnn-guest");
    let signal_len = 1024;
    let conv_a_len = signal_len / 4;
    let conv_b_len = signal_len / 8;
    let signal = generate_random_bytes(signal_len);
    let weights_fc_a = generate_random_bytes(conv_a_len * signal_len);
    let weights_fc_b = generate_random_bytes(conv_b_len * conv_a_len);
    program.set_input(&signal);
    program.set_input(&weights_fc_a);
    program.set_input(&weights_fc_b);

    generate_proof_and_verify::<F, PCS>(program)
}

fn sha2chain<F, PCS>() -> Vec<(tracing::Span, Box<dyn FnOnce()>)>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    let mut program = host::Program::new("sha2-chain-guest");
    program.set_input(&[5u8; 32]);
    program.set_input(&1024u32);

    generate_proof_and_verify::<F, PCS>(program)
}
