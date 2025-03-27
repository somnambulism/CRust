use crate::library::assembly::{FunctionDefinition, Instruction, Operand, Program, Reg};

fn fixup_instruction(instruction: Instruction) -> Vec<Instruction> {
    if let Instruction::Mov(src @ Operand::Stack(_), dst @ Operand::Stack(_)) = instruction {
        return vec![
            Instruction::Mov(src, Operand::Reg(Reg::R10)),
            Instruction::Mov(Operand::Reg(Reg::R10), dst),
        ];
    }
    vec![instruction]
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
