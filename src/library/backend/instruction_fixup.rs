use crate::library::assembly::{
    BinaryOperator, FunctionDefinition, Instruction, Operand, Program, Reg,
};

fn fixup_instruction(instruction: Instruction) -> Vec<Instruction> {
    match instruction {
        // Mov can't move a value from one memory address to another
        Instruction::Mov(src @ Operand::Stack(_), dst @ Operand::Stack(_)) => vec![
            Instruction::Mov(src, Operand::Reg(Reg::R10)),
            Instruction::Mov(Operand::Reg(Reg::R10), dst),
        ],
        // Idiv can't operate on constants
        Instruction::Idiv(Operand::Imm(i)) => vec![
            Instruction::Mov(Operand::Imm(i), Operand::Reg(Reg::R10)),
            Instruction::Idiv(Operand::Reg(Reg::R10)),
        ],
        // Add/Sub/And/Or/Xor can't use memory addresses for both operands
        Instruction::Binary {
            op:
                op @ BinaryOperator::Add
                | op @ BinaryOperator::Sub
                | op @ BinaryOperator::And
                | op @ BinaryOperator::Or
                | op @ BinaryOperator::Xor,
            src: src @ Operand::Stack(_),
            dst: dst @ Operand::Stack(_),
        } => vec![
            Instruction::Mov(src, Operand::Reg(Reg::R10)),
            Instruction::Binary {
                op,
                src: Operand::Reg(Reg::R10),
                dst,
            },
        ],
        // Destination of Mult can't be in memory
        Instruction::Binary {
            op: op @ BinaryOperator::Mult,
            src,
            dst: dst @ Operand::Stack(_),
        } => vec![
            Instruction::Mov(dst.clone(), Operand::Reg(Reg::R11)),
            Instruction::Binary {
                op,
                src,
                dst: Operand::Reg(Reg::R11),
            },
            Instruction::Mov(Operand::Reg(Reg::R11), dst),
        ],
        // Sal/Sar can't use memory addresses for both operands
        Instruction::Binary {
            op: op @ BinaryOperator::Sal | op @ BinaryOperator::Sar,
            src,
            dst: dst @ Operand::Stack(_),
        } => {
            // if the source is a constant, sal/sar can be done directly
            if let Operand::Imm(_) = src {
                return vec![Instruction::Binary { op, src, dst }];
            // otherwise, we need to move the destination to a register first
            } else {
                return vec![
                    Instruction::Mov(dst.clone(), Operand::Reg(Reg::R11)),
                    Instruction::Mov(src.clone(), Operand::Reg(Reg::CX)),
                    Instruction::Binary {
                        op,
                        src: Operand::Reg(Reg::CX),
                        dst: Operand::Reg(Reg::R11),
                    },
                    Instruction::Mov(Operand::Reg(Reg::R11), dst),
                ];
            }
        }
        // Both operands of cmp can't be in memory
        Instruction::Cmp(src @ Operand::Stack(_), dst @ Operand::Stack(_)) => vec![
            Instruction::Mov(src, Operand::Reg(Reg::R10)),
            Instruction::Cmp(Operand::Reg(Reg::R10), dst),
        ],
        // Second operand of cmp can't be a constant
        Instruction::Cmp(src, Operand::Imm(i)) => vec![
            Instruction::Mov(Operand::Imm(i), Operand::Reg(Reg::R11)),
            Instruction::Cmp(src, Operand::Reg(Reg::R11)),
        ],
        other => vec![other],
    }
}

fn fixup_function(last_stack_slot: i32, func: FunctionDefinition) -> FunctionDefinition {
    FunctionDefinition {
        name: func.name,
        instructions: std::iter::once(Instruction::AllocateStack(-last_stack_slot))
            .chain(func.instructions.into_iter().flat_map(fixup_instruction))
            .collect(),
    }
}

pub fn fixup_program(last_stack_slot: i32, program: Program) -> Program {
    Program {
        function: fixup_function(last_stack_slot, program.function),
    }
}
