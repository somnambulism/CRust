use std::collections::HashMap;

use crate::library::assembly::{FunctionDefinition, Instruction, Operand, Program};

#[derive(Debug)]
struct ReplacementState {
    current_offset: i32,
    offset_map: HashMap<String, i32>,
}

impl ReplacementState {
    fn new() -> Self {
        Self {
            current_offset: 0,
            offset_map: HashMap::new(),
        }
    }

    fn replace_operand(&mut self, operand: Operand) -> Operand {
        match operand {
            Operand::Pseudo(name) => {
                if let Some(&offset) = self.offset_map.get(&name) {
                    Operand::Stack(offset)
                } else {
                    let new_offset = self.current_offset - 4;
                    self.offset_map.insert(name, new_offset);
                    self.current_offset = new_offset;
                    Operand::Stack(new_offset)
                }
            }
            other => other,
        }
    }

    fn replace_pseudos_in_instruction(&mut self, instruction: Instruction) -> Instruction {
        match instruction {
            Instruction::Mov(src, dst) => {
                let new_src = self.replace_operand(src);
                let new_dst = self.replace_operand(dst);
                Instruction::Mov(new_src, new_dst)
            }
            Instruction::Unary(op, dst) => {
                let new_dst = self.replace_operand(dst);
                Instruction::Unary(op, new_dst)
            }
            Instruction::Binary { op, src, dst } => {
                let new_src = self.replace_operand(src);
                let new_dst = self.replace_operand(dst);
                Instruction::Binary {
                    op,
                    src: new_src,
                    dst: new_dst,
                }
            }
            Instruction::Cmp(op1, op2) => {
                let new_op1 = self.replace_operand(op1);
                let new_op2 = self.replace_operand(op2);
                Instruction::Cmp(new_op1, new_op2)
            }
            Instruction::Idiv(op) => {
                let new_op = self.replace_operand(op);
                Instruction::Idiv(new_op)
            }
            Instruction::SetCC(code, op) => {
                let new_op = self.replace_operand(op);
                Instruction::SetCC(code, new_op)
            }
            Instruction::Ret
            | Instruction::Cdq
            | Instruction::Label(_)
            | Instruction::JmpCC(_, _)
            | Instruction::Jmp(_) => instruction,
            Instruction::AllocateStack(_) => {
                panic!("Internal error: AllocateStack shouldn't be present at this point")
            }
        }
    }
}

fn replace_pseudos_in_function(func: FunctionDefinition) -> (FunctionDefinition, i32) {
    let mut state = ReplacementState::new();
    let fixed_instructions = func
        .instructions
        .into_iter()
        .map(|instr| state.replace_pseudos_in_instruction(instr))
        .collect();
    (
        FunctionDefinition {
            name: func.name,
            instructions: fixed_instructions,
        },
        state.current_offset,
    )
}

pub fn replace_pseudos(program: Program) -> (Program, i32) {
    let (fixed_def, last_stack_slot) = replace_pseudos_in_function(program.function);
    (
        Program {
            function: fixed_def,
        },
        last_stack_slot,
    )
}
