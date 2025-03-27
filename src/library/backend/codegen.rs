use std::vec;

use crate::library::{
    assembly::{
        FunctionDefinition as AssemblyFunction, Instruction, Operand, Program as AssemblyProgram,
        Reg, UnaryOperator,
    },
    tacky::{
        FunctionDefinition, Instruction as TackyInstruction, Program, TackyVal,
        UnaryOperator as TackyUnaryOp,
    },
};

fn convert_val(val: TackyVal) -> Operand {
    match val {
        TackyVal::Constant(i) => Operand::Imm(i),
        TackyVal::Var(v) => Operand::Pseudo(v),
    }
}

fn convert_op(op: TackyUnaryOp) -> UnaryOperator {
    match op {
        TackyUnaryOp::Complement => UnaryOperator::Not,
        TackyUnaryOp::Negate => UnaryOperator::Neg,
    }
}

fn convert_instruction(instruction: TackyInstruction) -> Vec<Instruction> {
    match instruction {
        TackyInstruction::Return(tacky_val) => {
            let asm_val = convert_val(tacky_val);
            vec![
                Instruction::Mov(asm_val, Operand::Reg(Reg::AX)),
                Instruction::Ret,
            ]
        }
        TackyInstruction::Unary { op, src, dst } => {
            let asm_op = convert_op(op);
            let asm_src = convert_val(src);
            let asm_dst = convert_val(dst);
            vec![
                Instruction::Mov(asm_src, asm_dst.clone()),
                Instruction::Unary(asm_op, asm_dst),
            ]
        }
    }
}

fn convert_function(function_definition: FunctionDefinition) -> AssemblyFunction {
    let instructions = function_definition
        .body
        .into_iter()
        .flat_map(convert_instruction)
        .collect();
    AssemblyFunction {
        name: function_definition.name,
        instructions,
    }
}

pub fn generate(program: Program) -> AssemblyProgram {
    AssemblyProgram {
        function: convert_function(program.function),
    }
}
