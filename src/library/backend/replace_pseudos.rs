use std::collections::HashMap;

use crate::library::{
    assembly::{FunctionDefinition, Instruction, Operand, Program},
    symbols::SymbolTable,
};

#[derive(Debug)]
struct ReplacementState {
    current_offset: isize,
    offset_map: HashMap<String, isize>,
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
            Instruction::Push(op) => {
                let new_op = self.replace_operand(op);
                Instruction::Push(new_op)
            }
            Instruction::Ret
            | Instruction::Cdq
            | Instruction::Label(_)
            | Instruction::JmpCC(_, _)
            | Instruction::Jmp(_)
            | Instruction::DeallocateStack(_)
            | Instruction::Call(_)
            | Instruction::AllocateStack(_) => instruction,
        }
    }
}

fn replace_pseudos_in_function(
    func: FunctionDefinition,
    symbol_table: &mut SymbolTable,
) -> FunctionDefinition {
    let mut state = ReplacementState::new();
    let fixed_instructions = func
        .instructions
        .into_iter()
        .map(|instr| state.replace_pseudos_in_instruction(instr))
        .collect();
    symbol_table.set_bytes_required(&func.name, state.current_offset);
    FunctionDefinition {
        name: func.name,
        instructions: fixed_instructions,
    }
}

pub fn replace_pseudos(program: Program, symbol_table: &mut SymbolTable) -> Program {
    let fixed_defs = program
        .function
        .into_iter()
        .map(|fn_def| replace_pseudos_in_function(fn_def, symbol_table))
        .collect::<Vec<_>>();
    Program {
        function: fixed_defs,
    }
}
