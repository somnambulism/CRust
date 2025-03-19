use std::vec;

use super::{
    assembly::{
        FunctionDefinition as AssemblyFunction, Instruction, Operand, Program as AssemblyProgram,
    },
    ast::{Exp, FunctionDefinition, Program, Statement},
};

fn convert_exp(i: Exp) -> Operand {
    match i {
        Exp::Constant(i) => Operand::Imm(i),
    }
}

fn convert_statement(e: Statement) -> Vec<Instruction> {
    match e {
        Statement::Return(e) => {
            let v = convert_exp(e);
            vec![Instruction::Mov(v, Operand::Register), Instruction::Ret]
        }
    }
}

fn convert_function(function_definition: FunctionDefinition) -> AssemblyFunction {
    AssemblyFunction {
        name: function_definition.name,
        instructions: convert_statement(function_definition.body),
    }
}

pub fn generate(fn_def: Program) -> AssemblyProgram {
    AssemblyProgram {
        function: convert_function(fn_def.function),
    }
}
