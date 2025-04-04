use std::vec;

use crate::library::{
    assembly::{
        BinaryOperator, CondCode, FunctionDefinition as AssemblyFunction, Instruction, Operand,
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
        TackyUnaryOp::Not => {
            panic!("Internal error, can't convert TACKY 'not' directly to assembly")
        }
    }
}

fn convert_binop(op: TackyBinaryOp) -> BinaryOperator {
    match op {
        TackyBinaryOp::Add => BinaryOperator::Add,
        TackyBinaryOp::Subtract => BinaryOperator::Sub,
        TackyBinaryOp::Multiply => BinaryOperator::Mult,
        TackyBinaryOp::BitwiseAnd => BinaryOperator::And,
        TackyBinaryOp::BitwiseOr => BinaryOperator::Or,
        TackyBinaryOp::Xor => BinaryOperator::Xor,
        TackyBinaryOp::LeftShift => BinaryOperator::Sal,
        TackyBinaryOp::RightShift => BinaryOperator::Sar,
        TackyBinaryOp::Divide
        | TackyBinaryOp::Mod
        | TackyBinaryOp::Equal
        | TackyBinaryOp::NotEqual
        | TackyBinaryOp::GreaterOrEqual
        | TackyBinaryOp::LessOrEqual
        | TackyBinaryOp::GreaterThan
        | TackyBinaryOp::LessThan => {
            panic!("Internal error: not a binary assembly instruction")
        }
    }
}

fn convert_cond_code(cond_code: TackyBinaryOp) -> CondCode {
    match cond_code {
        TackyBinaryOp::Equal => CondCode::E,
        TackyBinaryOp::NotEqual => CondCode::NE,
        TackyBinaryOp::GreaterThan => CondCode::G,
        TackyBinaryOp::GreaterOrEqual => CondCode::GE,
        TackyBinaryOp::LessThan => CondCode::L,
        TackyBinaryOp::LessOrEqual => CondCode::LE,
        _ => panic!("Internal error: not a condition code"),
    }
}

fn convert_instruction(instruction: TackyInstruction) -> Vec<Instruction> {
    match instruction {
        TackyInstruction::Copy { src, dst } => {
            let asm_src = convert_val(src);
            let asm_dst = convert_val(dst);
            vec![Instruction::Mov(asm_src, asm_dst)]
        }
        TackyInstruction::Return(tacky_val) => {
            let asm_val = convert_val(tacky_val);
            vec![
                Instruction::Mov(asm_val, Operand::Reg(Reg::AX)),
                Instruction::Ret,
            ]
        }
        TackyInstruction::Unary {
            op: TackyUnaryOp::Not,
            src,
            dst,
        } => {
            let asm_src = convert_val(src);
            let asm_dst = convert_val(dst);
            vec![
                Instruction::Cmp(Operand::Imm(0), asm_src),
                Instruction::Mov(Operand::Imm(0), asm_dst.clone()),
                Instruction::SetCC(CondCode::E, asm_dst),
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
                // Relational operator
                TackyBinaryOp::Equal
                | TackyBinaryOp::NotEqual
                | TackyBinaryOp::GreaterThan
                | TackyBinaryOp::GreaterOrEqual
                | TackyBinaryOp::LessThan
                | TackyBinaryOp::LessOrEqual => {
                    let cond_code = convert_cond_code(op);
                    vec![
                        Instruction::Cmp(asm_src2, asm_src1),
                        Instruction::Mov(Operand::Imm(0), asm_dst.clone()),
                        Instruction::SetCC(cond_code, asm_dst),
                    ]
                }
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
                // Bitwise shift
                TackyBinaryOp::LeftShift | TackyBinaryOp::RightShift => {
                    vec![
                        // Instruction::Mov(asm_src2, Operand::Reg(Reg::CX)),
                        Instruction::Mov(asm_src1, asm_dst.clone()),
                        Instruction::Binary {
                            op: convert_binop(op),
                            src: asm_src2,
                            dst: asm_dst,
                        },
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
        TackyInstruction::Jump(target) => vec![Instruction::Jmp(target)],
        TackyInstruction::JumpIfZero(cond, target) => {
            let asm_cond = convert_val(cond);
            vec![
                Instruction::Cmp(Operand::Imm(0), asm_cond),
                Instruction::JmpCC(CondCode::E, target),
            ]
        }
        TackyInstruction::JumpIfNotZero(cond, target) => {
            let asm_cond = convert_val(cond);
            vec![
                Instruction::Cmp(Operand::Imm(0), asm_cond),
                Instruction::JmpCC(CondCode::NE, target),
            ]
        }
        TackyInstruction::Label(l) => vec![Instruction::Label(l)],
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
