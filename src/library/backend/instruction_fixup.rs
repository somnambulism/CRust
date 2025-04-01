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
        // Add/Sub can't use memory addresses for both operands
        Instruction::Binary {
            op: op @ BinaryOperator::Add | op @ BinaryOperator::Sub,
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
            op: BinaryOperator::Mult,
            src,
            dst: dst @ Operand::Stack(_),
        } => vec![
            Instruction::Mov(dst.clone(), Operand::Reg(Reg::R11)),
            Instruction::Binary {
                op: BinaryOperator::Mult,
                src,
                dst: Operand::Reg(Reg::R11),
            },
            Instruction::Mov(Operand::Reg(Reg::R11), dst),
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
