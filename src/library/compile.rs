use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::library::{emit::CodeEmitter, lex::Lexer, parse::Parser, settings::current_platform};

use super::{codegen, settings::Stage};

pub fn compile(stage: &Stage, src_file: &str) {
    // Read file
    let source = fs::read_to_string(src_file).expect("Failed to read source file");

    // Lexical analysis
    let lexer = Lexer::new();
    let lexed_tokens = lexer
        .lex(&source)
        .unwrap_or_else(|e| panic!("Error: {}", e));

    if *stage == Stage::Lex {
        return;
    }

    let mut parser = Parser::new(lexed_tokens);
    let ast = parser
        .parse_program()
        .unwrap_or_else(|e| panic!("Error: {}", e));

    if *stage == Stage::Parse {
        return;
    }

    let asm_ast = codegen::generate(ast);

    if *stage == Stage::Codegen {
        return;
    }

    let mut asm_filename = PathBuf::from(src_file);
    asm_filename.set_extension("s");
    println!("Emitting assembly to {}", asm_filename.to_string_lossy());
    let mut emitter =
        CodeEmitter::new(current_platform(), &asm_filename.to_string_lossy()).unwrap();
    emitter.emit(&asm_ast).unwrap();
}
