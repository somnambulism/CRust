use std::vec;

use crate::library::{
    assembly::{
        BinaryOperator, FunctionDefinition as AssemblyFunction, Instruction, Operand,
        Program as AssemblyProgram, Reg, UnaryOperator,
    },
    tacky::{
        BinaryOperator as TackyBinaryOp, FunctionDefinition, Instruction as TackyInstruction,
        Program, TackyVal, UnaryOperator as TackyUnaryOp,
    },
};

fn convert_val(val: TackyVal) -> Operand {
    match val {
        TackyVal::Constant(i) => Operand::Imm(i),
        TackyVal::Var(v) => Operand::Pseudo(v),
    }
}

fn convert_unop(op: TackyUnaryOp) -> UnaryOperator {
    match op {
        TackyUnaryOp::Complement => UnaryOperator::Not,
        TackyUnaryOp::Negate => UnaryOperator::Neg,
    }
}

fn convert_binop(op: TackyBinaryOp) -> BinaryOperator {
    match op {
        TackyBinaryOp::Add => BinaryOperator::Add,
        TackyBinaryOp::Subtract => BinaryOperator::Sub,
        TackyBinaryOp::Multiply => BinaryOperator::Mult,
        TackyBinaryOp::Divide | TackyBinaryOp::Mod => {
            panic!("Internal error: shouldn't handle division like other binary operators")
        }
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
            let asm_op = convert_unop(op);
            let asm_src = convert_val(src);
            let asm_dst = convert_val(dst);
            vec![
                Instruction::Mov(asm_src, asm_dst.clone()),
                Instruction::Unary(asm_op, asm_dst),
            ]
        }
        TackyInstruction::Binary {
            op,
            src1,
            src2,
            dst,
        } => {
            let asm_src1 = convert_val(src1);
            let asm_src2 = convert_val(src2);
            let asm_dst = convert_val(dst);
            match op {
                // Division/modulo
                TackyBinaryOp::Divide | TackyBinaryOp::Mod => {
                    let result_reg = if op == TackyBinaryOp::Divide {
                        Reg::AX
                    } else {
                        Reg::DX
                    };
                    vec![
                        Instruction::Mov(asm_src1, Operand::Reg(Reg::AX)),
                        Instruction::Cdq,
                        Instruction::Idiv(asm_src2),
                        Instruction::Mov(Operand::Reg(result_reg), asm_dst),
                    ]
                }
                // Addition/subtraction/multiplication
                _ => {
                    let asm_op = convert_binop(op);
                    vec![
                        Instruction::Mov(asm_src1, asm_dst.clone()),
                        Instruction::Binary {
                            op: asm_op,
                            src: asm_src2,
                            dst: asm_dst,
                        },
                    ]
                }
            }
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
