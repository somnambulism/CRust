mod library;

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

use library::{compile, settings::Stage};

fn validate_extension(filename: &str) {
    let ext = Path::new(filename)
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("default");

    if ext != "c" && ext != "h" {
        panic!("Expected C source file with .c or .h extension");
    }
}

fn replace_extension(filename: &str, new_extension: &str) -> String {
    let mut asm_filename = PathBuf::from(filename);
    asm_filename.set_extension(new_extension);
    format!("{}", asm_filename.to_string_lossy())
}

fn run_command(cmd: &str, args: &[&str]) {
    let status = Command::new(cmd)
        .args(args)
        .status()
        .expect("Failed to execute command");

    if !status.success() {
        panic!("Command failed: {}", cmd);
    }
}

fn preprocess(src: &str) -> String {
    validate_extension(src);
    let output = replace_extension(src, "i");
    run_command("gcc", &["-E", "-P", src, "-o", &output]);
    output
}

fn compile(stage: &Stage, preprocessed_src: &str, debug: bool) -> String {
    compile::compile(&stage, preprocessed_src, debug);
    run_command("rm", &[preprocessed_src]);
    replace_extension(preprocessed_src, "s")
}

fn assemble_and_link(src: &str, cleanup: bool) {
    let assembly_file = replace_extension(src, "s");
    let output_file = replace_extension(src, "exe");

    run_command("gcc", &[&assembly_file, "-o", &output_file]);

    if cleanup {
        run_command("rm", &[&assembly_file]);
    }
}

fn driver(debug: bool, stage: &Stage, src: &str) {
    let preprocessed_name = preprocess(src);
    let assembly_name = compile(stage, &preprocessed_name, debug);

    if *stage == Stage::Executable {
        assemble_and_link(&assembly_name, !debug);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <source-file>", args[0]);
        std::process::exit(1);
    }

    let src_file = &args[1];

    let stage = if args.contains(&"--lex".to_string()) {
        Stage::Lex
    } else if args.contains(&"--parse".to_string()) {
        Stage::Parse
    } else if args.contains(&"--tacky".to_string()) {
        Stage::Tacky
    } else if args.contains(&"--codegen".to_string()) {
        Stage::Codegen
    } else if args.contains(&"-S".to_string()) || args.contains(&"-s".to_string()) {
        Stage::Assembly
    } else {
        Stage::Executable
    };
    let debug = args.contains(&"-d".to_string());

    driver(debug, &stage, src_file);
}
