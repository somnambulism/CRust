use std::i32;

use crate::library::{
    assembly::{AsmType, BinaryOperator, Instruction, Operand, Program, Reg, TopLevel},
    backend::assembly_symbols::SymbolTable,
    util::rounding::round_away_from_zero,
};

const INT32_MAX: i64 = i32::MAX as i64;
const INT32_MIN: i64 = i32::MIN as i64;

fn is_large(imm: i64) -> bool {
    imm > INT32_MAX || imm < INT32_MIN
}

fn is_larger_than_uint(imm: i64) -> bool {
    // use unsigned upper-bound for positives
    let max_i = 0xFFFFFFFF; // 2^32 - 1
    // use signed 32-bit lower bound for negatives
    imm > max_i || imm < INT32_MIN
}

fn is_constant(operand: &Operand) -> bool {
    matches!(operand, Operand::Imm(_))
}

fn is_memory(operand: &Operand) -> bool {
    matches!(operand, Operand::Stack(_) | Operand::Data(_))
}

fn fixup_instruction(instruction: Instruction) -> Vec<Instruction> {
    match instruction {
        // Mov can't move a value from one memory address to another
        Instruction::Mov(
            t,
            src @ Operand::Stack(_) | src @ Operand::Data(_),
            dst @ Operand::Stack(_) | dst @ Operand::Data(_),
        ) => {
            let scratch = if t == AsmType::Double {
                Operand::Reg(Reg::XMM14)
            } else {
                Operand::Reg(Reg::R10)
            };
            vec![
                Instruction::Mov(t.clone(), src, scratch.clone()),
                Instruction::Mov(t, scratch, dst),
            ]
        }
        // Mov can't move a large constant to a memory address
        Instruction::Mov(
            AsmType::Quadword,
            src @ Operand::Imm(i),
            dst @ Operand::Stack(_) | dst @ Operand::Data(_),
        ) if is_large(i) => {
            vec![
                Instruction::Mov(AsmType::Quadword, src, Operand::Reg(Reg::R10)),
                Instruction::Mov(AsmType::Quadword, Operand::Reg(Reg::R10), dst),
            ]
        }
        // Moving a quadword-size constant with a longword operand size produces
        // assembler warning
        Instruction::Mov(AsmType::Longword, Operand::Imm(i), dst) if is_larger_than_uint(i) => {
            // reduce modulo 2^32 by zeroing out upper 32 bits
            let bitmask = 0xffffffff;
            let reduced = i & bitmask;
            vec![Instruction::Mov(
                AsmType::Longword,
                Operand::Imm(reduced),
                dst,
            )]
        }
        // Movsx can't handle immediate source or memory dst
        Instruction::Movsx(
            src @ Operand::Imm(_),
            dst @ Operand::Stack(_) | dst @ Operand::Data(_),
        ) => {
            vec![
                Instruction::Mov(AsmType::Longword, src, Operand::Reg(Reg::R10)),
                Instruction::Movsx(Operand::Reg(Reg::R10), Operand::Reg(Reg::R11)),
                Instruction::Mov(AsmType::Quadword, Operand::Reg(Reg::R11), dst),
            ]
        }
        Instruction::Movsx(src @ Operand::Imm(_), dst) => {
            vec![
                Instruction::Mov(AsmType::Longword, src, Operand::Reg(Reg::R10)),
                Instruction::Movsx(Operand::Reg(Reg::R10), dst),
            ]
        }
        Instruction::Movsx(src, dst @ Operand::Stack(_) | dst @ Operand::Data(_)) => {
            vec![
                Instruction::Movsx(src, Operand::Reg(Reg::R11)),
                Instruction::Mov(AsmType::Quadword, Operand::Reg(Reg::R11), dst),
            ]
        }
        // Rewrite MovZeroExtend as one or two Mov instructions
        Instruction::MovZeroExtend(src, dst) => match dst {
            Operand::Reg(_) => vec![Instruction::Mov(AsmType::Longword, src, dst)],
            _ => {
                vec![
                    Instruction::Mov(AsmType::Longword, src, Operand::Reg(Reg::R11)),
                    Instruction::Mov(AsmType::Quadword, Operand::Reg(Reg::R11), dst),
                ]
            }
        },
        // Idiv can't operate on constants
        Instruction::Idiv(t, Operand::Imm(i)) => vec![
            Instruction::Mov(t.clone(), Operand::Imm(i), Operand::Reg(Reg::R10)),
            Instruction::Idiv(t, Operand::Reg(Reg::R10)),
        ],
        Instruction::Div(t, Operand::Imm(i)) => vec![
            Instruction::Mov(t.clone(), Operand::Imm(i), Operand::Reg(Reg::R10)),
            Instruction::Div(t, Operand::Reg(Reg::R10)),
        ],
        // Binary operations on double require register as destination
        Instruction::Binary {
            t: AsmType::Double,
            dst: Operand::Reg(_),
            ..
        } => vec![instruction],
        Instruction::Binary {
            op,
            t: AsmType::Double,
            src,
            dst,
        } => vec![
            Instruction::Mov(AsmType::Double, dst.clone(), Operand::Reg(Reg::XMM15)),
            Instruction::Binary {
                op,
                t: AsmType::Double,
                src,
                dst: Operand::Reg(Reg::XMM15),
            },
            Instruction::Mov(AsmType::Double, Operand::Reg(Reg::XMM15), dst),
        ],
        // Add/Sub/And/Or/Xor can't use take large immediates as source operands
        Instruction::Binary {
            op:
                op @ BinaryOperator::Add
                | op @ BinaryOperator::Sub
                | op @ BinaryOperator::And
                | op @ BinaryOperator::Or
                | op @ BinaryOperator::Xor,
            t: AsmType::Quadword,
            src: src @ Operand::Imm(i),
            dst,
        } if is_large(i) => {
            vec![
                Instruction::Mov(AsmType::Quadword, src, Operand::Reg(Reg::R10)),
                Instruction::Binary {
                    op,
                    t: AsmType::Quadword,
                    src: Operand::Reg(Reg::R10),
                    dst,
                },
            ]
        }
        // Add/Sub/And/Or/Xor can't use memory addresses for both operands
        Instruction::Binary {
            op:
                op @ BinaryOperator::Add
                | op @ BinaryOperator::Sub
                | op @ BinaryOperator::And
                | op @ BinaryOperator::Or
                | op @ BinaryOperator::Xor,
            t,
            src: src @ Operand::Stack(_) | src @ Operand::Data(_),
            dst: dst @ Operand::Stack(_) | dst @ Operand::Data(_),
        } => vec![
            Instruction::Mov(t.clone(), src, Operand::Reg(Reg::R10)),
            Instruction::Binary {
                op,
                t,
                src: Operand::Reg(Reg::R10),
                dst,
            },
        ],
        // Destination of Mult can't be in memory; src can't be a big operand
        Instruction::Binary {
            op: BinaryOperator::Mult,
            t: AsmType::Quadword,
            src: src @ Operand::Imm(i),
            dst: dst @ Operand::Stack(_) | dst @ Operand::Data(_),
        } if is_large(i) => {
            // rewrite both operands
            vec![
                Instruction::Mov(AsmType::Quadword, src, Operand::Reg(Reg::R10)),
                Instruction::Mov(AsmType::Quadword, dst.clone(), Operand::Reg(Reg::R11)),
                Instruction::Binary {
                    op: BinaryOperator::Mult,
                    t: AsmType::Quadword,
                    src: Operand::Reg(Reg::R10),
                    dst: Operand::Reg(Reg::R11),
                },
                Instruction::Mov(AsmType::Quadword, Operand::Reg(Reg::R11), dst),
            ]
        }
        Instruction::Binary {
            op: BinaryOperator::Mult,
            t: AsmType::Quadword,
            src: src @ Operand::Imm(i),
            dst,
        } if is_large(i) => {
            // just rewrite src
            vec![
                Instruction::Mov(AsmType::Quadword, src, Operand::Reg(Reg::R10)),
                Instruction::Binary {
                    op: BinaryOperator::Mult,
                    t: AsmType::Quadword,
                    src: Operand::Reg(Reg::R10),
                    dst,
                },
            ]
        }
        Instruction::Binary {
            op: BinaryOperator::Mult,
            t,
            src,
            dst: dst @ Operand::Stack(_) | dst @ Operand::Data(_),
        } => vec![
            Instruction::Mov(t.clone(), dst.clone(), Operand::Reg(Reg::R11)),
            Instruction::Binary {
                op: BinaryOperator::Mult,
                t: t.clone(),
                src,
                dst: Operand::Reg(Reg::R11),
            },
            Instruction::Mov(t, Operand::Reg(Reg::R11), dst),
        ],
        // Sal/Sar can't use memory addresses for both operands
        Instruction::Binary {
            op:
                op @ BinaryOperator::Sal
                | op @ BinaryOperator::Sar
                | op @ BinaryOperator::Shl
                | op @ BinaryOperator::Shr,
            t,
            src,
            dst: dst @ Operand::Stack(_) | dst @ Operand::Data(_),
        } => {
            // if the source is a constant, sal/sar can be done directly
            if let Operand::Imm(_) = src {
                return vec![Instruction::Binary { op, t, src, dst }];
            // otherwise, we need to move the destination to a register first
            } else {
                return vec![
                    Instruction::Mov(t.clone(), dst.clone(), Operand::Reg(Reg::R11)),
                    Instruction::Mov(t.clone(), src.clone(), Operand::Reg(Reg::CX)),
                    Instruction::Binary {
                        op,
                        t: t.clone(),
                        src: Operand::Reg(Reg::CX),
                        dst: Operand::Reg(Reg::R11),
                    },
                    Instruction::Mov(t, Operand::Reg(Reg::R11), dst),
                ];
            }
        }
        // Destination of comisd must be a register
        Instruction::Cmp(AsmType::Double, _, Operand::Reg(_)) => vec![instruction],
        Instruction::Cmp(AsmType::Double, src, dst) => vec![
            Instruction::Mov(AsmType::Double, dst, Operand::Reg(Reg::XMM15)),
            Instruction::Cmp(AsmType::Double, src, Operand::Reg(Reg::XMM15)),
        ],
        // Both operands of cmp can't be in memory
        Instruction::Cmp(
            t,
            src @ Operand::Stack(_) | src @ Operand::Data(_),
            dst @ Operand::Stack(_) | dst @ Operand::Data(_),
        ) => vec![
            Instruction::Mov(t.clone(), src, Operand::Reg(Reg::R10)),
            Instruction::Cmp(t, Operand::Reg(Reg::R10), dst),
        ],
        // first operand of Cmp can't be a large constant, second can't be a constant
        // at all
        Instruction::Cmp(AsmType::Quadword, src @ Operand::Imm(i), dst @ Operand::Imm(_))
            if is_large(i) =>
        {
            vec![
                Instruction::Mov(AsmType::Quadword, src, Operand::Reg(Reg::R10)),
                Instruction::Mov(AsmType::Quadword, dst, Operand::Reg(Reg::R11)),
                Instruction::Cmp(
                    AsmType::Quadword,
                    Operand::Reg(Reg::R10),
                    Operand::Reg(Reg::R11),
                ),
            ]
        }
        Instruction::Cmp(AsmType::Quadword, src @ Operand::Imm(i), dst) if is_large(i) => vec![
            Instruction::Mov(AsmType::Quadword, src, Operand::Reg(Reg::R10)),
            Instruction::Cmp(AsmType::Quadword, Operand::Reg(Reg::R10), dst),
        ],
        // Second operand of cmp can't be a constant
        Instruction::Cmp(t, src, Operand::Imm(i)) => vec![
            Instruction::Mov(t.clone(), Operand::Imm(i), Operand::Reg(Reg::R11)),
            Instruction::Cmp(t, src, Operand::Reg(Reg::R11)),
        ],
        Instruction::Push(src @ Operand::Imm(i)) if is_large(i) => vec![
            Instruction::Mov(AsmType::Quadword, src, Operand::Reg(Reg::R10)),
            Instruction::Push(Operand::Reg(Reg::R10)),
        ],
        // destination of cvttsd2si must be a register
        Instruction::Cvttsd2si(t, src, dst @ Operand::Stack(_) | dst @ Operand::Data(_)) => vec![
            Instruction::Cvttsd2si(t.clone(), src, Operand::Reg(Reg::R11)),
            Instruction::Mov(t, Operand::Reg(Reg::R11), dst),
        ],
        Instruction::Cvtsi2sd(t, src, dst) => {
            if is_constant(&src) && is_memory(&dst) {
                vec![
                    Instruction::Mov(t.clone(), src, Operand::Reg(Reg::R10)),
                    Instruction::Cvtsi2sd(t, Operand::Reg(Reg::R10), Operand::Reg(Reg::XMM15)),
                    Instruction::Mov(AsmType::Double, Operand::Reg(Reg::XMM15), dst),
                ]
            } else if is_constant(&src) {
                vec![
                    Instruction::Mov(t.clone(), src, Operand::Reg(Reg::R10)),
                    Instruction::Cvtsi2sd(t, Operand::Reg(Reg::R10), dst),
                ]
            } else if is_memory(&dst) {
                vec![
                    Instruction::Cvtsi2sd(t, src, Operand::Reg(Reg::XMM15)),
                    Instruction::Mov(AsmType::Double, Operand::Reg(Reg::XMM15), dst),
                ]
            } else {
                vec![Instruction::Cvtsi2sd(t, src, dst)]
            }
        }
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
            let stack_bytes = round_away_from_zero(16, -(symbol_table.get_bytes_required(&name)));
            let stack_bytes_op = Operand::Imm(stack_bytes as i64);
            let x = TopLevel::Function {
                name,
                global,
                instructions: std::iter::once(Instruction::Binary {
                    op: BinaryOperator::Sub,
                    t: AsmType::Quadword,
                    src: stack_bytes_op,
                    dst: Operand::Reg(Reg::SP),
                })
                .chain(instructions.into_iter().flat_map(fixup_instruction))
                .collect(),
            };
            x
        }
        TopLevel::StaticVariable { .. } | TopLevel::StaticConstant { .. } => tl,
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
