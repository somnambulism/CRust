use crate::library::{
    assembly::{BinaryOperator, Instruction, Operand, Program, Reg, TopLevel},
    symbols::SymbolTable,
    util::rounding::round_away_from_zero,
};

fn fixup_instruction(instruction: Instruction) -> Vec<Instruction> {
    match instruction {
        // Mov can't move a value from one memory address to another
        Instruction::Mov(
            src @ Operand::Stack(_) | src @ Operand::Data(_),
            dst @ Operand::Stack(_) | dst @ Operand::Data(_),
        ) => {
            vec![
                Instruction::Mov(src, Operand::Reg(Reg::R10)),
                Instruction::Mov(Operand::Reg(Reg::R10), dst),
            ]
        }
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
            src: src @ Operand::Stack(_) | src @ Operand::Data(_),
            dst: dst @ Operand::Stack(_) | dst @ Operand::Data(_),
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
            dst: dst @ Operand::Stack(_) | dst @ Operand::Data(_),
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
            dst: dst @ Operand::Stack(_) | dst @ Operand::Data(_),
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
        Instruction::Cmp(
            src @ Operand::Stack(_) | src @ Operand::Data(_),
            dst @ Operand::Stack(_) | dst @ Operand::Data(_),
        ) => vec![
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

fn fixup_tl(tl: TopLevel, symbol_table: &SymbolTable) -> TopLevel {
    match tl {
        TopLevel::Function {
            name,
            global,
            instructions,
        } => {
            let stack_bytes = -symbol_table.get_bytes_required(&name);
            let x = TopLevel::Function {
                name,
                global,
                instructions: std::iter::once(Instruction::AllocateStack(
                    round_away_from_zero(16, stack_bytes).try_into().unwrap(),
                ))
                .chain(instructions.into_iter().flat_map(fixup_instruction))
                .collect(),
            };
            x
        }
        TopLevel::StaticVariable { .. } => tl,
    }
}

pub fn fixup_program(program: Program, symbol_table: &SymbolTable) -> Program {
    let fixed_functions = program
        .top_levels
        .into_iter()
        .map(|tl| fixup_tl(tl, symbol_table))
        .collect();
    Program {
        top_levels: fixed_functions,
    }
}
