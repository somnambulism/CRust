use std::io::Write;
use std::{
    fs::{self, File},
    path::PathBuf,
};

use crate::library::backend::codegen::CodeGen;
use crate::library::semantic_analysis::collect_switch_cases::analyze_switches;
use crate::library::tacky_gen::TackyGen;
use crate::library::{emit::CodeEmitter, lex::Lexer, parse::Parser, settings::current_platform};

use super::semantic_analysis::label_loops::LoopsLabeller;
use super::semantic_analysis::identifier_resolution::Resolver;
use super::semantic_analysis::typecheck::TypeChecker;
use super::semantic_analysis::validate_labels::validate_labels;
use super::{
    backend::{instruction_fixup, replace_pseudos},
    settings::Stage,
};

pub fn compile(stage: &Stage, src_file: &str, debug: bool) {
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

    // Semantic analysis
    // 1. Resolve identifiers
    let mut variable_resolver = Resolver::new();
    let ast = variable_resolver.resolve(ast);
    // 2. Label gotos
    let ast = validate_labels(ast).unwrap();
    // 3. Annotate loops and break/continue statements
    let mut loops_labeller = LoopsLabeller::new();
    let ast = loops_labeller.label_loops(ast);
    // 4. Typecheck definitions and uses of functions and variables
    let mut typeckecher = TypeChecker::new();
    let ast = typeckecher.typecheck(&ast);
    // 5. Collect cases in switch statements (need to typecheck first)
    let ast = analyze_switches(ast).unwrap();

    if *stage == Stage::Validate {
        return;
    }

    // Convert the AST to TACKY
    let mut tacky_generator = TackyGen::new(typeckecher.symbol_table);
    let tacky = tacky_generator.generate(ast);
    if debug {
        let mut tacky_filename = PathBuf::from(src_file);
        tacky_filename.set_extension("debug.tacky");
        let mut writer = File::create(tacky_filename).unwrap();
        let _ = writeln!(writer, "{:#?}", tacky);
    }

    if *stage == Stage::Tacky {
        return;
    }

    // 1. Convert TACKY to assembly
    let mut codegenerator = CodeGen::new(tacky_generator.symbol_table);
    let asm_ast = codegenerator.generate(&tacky);
    if debug {
        let mut prealloc_filename = PathBuf::from(src_file);
        prealloc_filename.set_extension("prealloc.debug.s");
        let mut writer = File::create(prealloc_filename).unwrap();
        let _ = writeln!(writer, "{:#?}", asm_ast);
    }
    // 2. Replace pseudoregisters with Stack operands
    let asm_ast1 = replace_pseudos::replace_pseudos(asm_ast, &mut codegenerator.assembly_symbols);
    // 3. Fixup instructions
    let asm_ast2 = instruction_fixup::fixup_program(asm_ast1, &mut codegenerator.assembly_symbols);

    if *stage == Stage::Codegen {
        return;
    }

    let mut asm_filename = PathBuf::from(src_file);
    asm_filename.set_extension("s");
    let mut emitter = CodeEmitter::new(
        current_platform(),
        &asm_filename.to_string_lossy(),
        codegenerator.assembly_symbols,
    )
    .unwrap();
    emitter.emit(&asm_ast2).unwrap();
}
