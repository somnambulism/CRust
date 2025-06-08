mod library;

use std::{
    path::{Path, PathBuf},
    process::Command,
};

use clap::Parser;
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

fn assemble_and_link(src: &str, link: bool, cleanup: bool) {
    let assembly_file = replace_extension(src, "s");
    let output_file = if link {
        replace_extension(src, "exe")
    } else {
        replace_extension(src, "o")
    };

    let mut args = vec![];
    if !link {
        args.push("-c");
    }
    args.extend(&[
        &assembly_file,
        "-o",
        &output_file,
    ]);

    run_command("gcc", &args);

    if cleanup {
        run_command("rm", &[&assembly_file]);
    }
}

fn driver(debug: bool, stage: &Stage, src: &str) {
    let preprocessed_name = preprocess(src);
    let assembly_name = compile(stage, &preprocessed_name, debug);

    match *stage {
        Stage::Executable => {
            assemble_and_link(&assembly_name, true, !debug);
        }
        Stage::Obj => {
            assemble_and_link(&assembly_name, false, !debug);
        }
        _ => (),
    }
}

// Command-line options
#[derive(Parser, Debug)]
#[command(about = "Simple C compiler")]
struct Args {
    #[arg(long, help = "Run the lexer")]
    lex: bool,

    #[arg(long, help = "Run the lexer and parser")]
    parse: bool,

    #[arg(long, help = "Run the lexer, parser, and semantic analysis")]
    validate: bool,

    #[arg(
        long,
        help = "Run the lexer, parser, semantic analysis, and tacky generator"
    )]
    tacky: bool,

    #[arg(
        long,
        help = "Run through code generation but stop before emitting assembly"
    )]
    codegen: bool,

    #[arg(
        short = 's',
        short_alias = 'S',
        help = "Stop before assembling (keep .s file)"
    )]
    assembly: bool,

    #[arg(short = 'c', help = "Stop before invoking linker (keep .o file)")]
    obj: bool,

    #[arg(
        short = 'd',
        help = "Write out pre- and post-register-allocation assembly and DOT files of \
     interference graphs."
    )]
    debug: bool,

    #[arg()]
    input: Option<String>,
}

fn main() {
    // Skip first argument (executable name)
    let args = Args::parse();

    let stage = if args.lex {
        Stage::Lex
    } else if args.parse {
        Stage::Parse
    } else if args.validate {
        Stage::Validate
    } else if args.tacky {
        Stage::Tacky
    } else if args.codegen {
        Stage::Codegen
    } else if args.assembly {
        Stage::Assembly
    } else if args.obj {
        Stage::Obj
    } else {
        Stage::Executable
    };

    if let Some(input_file) = args.input {
        driver(args.debug, &stage, &input_file);
    } else {
        eprintln!("Usage: <program> [options] <source-file>");
        std::process::exit(1);
    }
}
