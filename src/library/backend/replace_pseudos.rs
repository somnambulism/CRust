use std::collections::HashMap;

use crate::library::{
    assembly::{Instruction, Operand, Program, TopLevel},
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

    fn replace_operand(&mut self, operand: &Operand, symbols: &SymbolTable) -> Operand {
        match operand {
            Operand::Pseudo(s) => {
                if symbols.is_static(&s) {
                    return Operand::Data(s.to_string());
                } else {
                    // We've already assigned this operand a stack slot
                    if let Some(&offset) = self.offset_map.get(s) {
                        Operand::Stack(offset)
                    // We haven't already assigned it a stack slot;
                    // assign it and update state
                    } else {
                        let new_offset = self.current_offset - 4;
                        self.offset_map.insert(s.to_string(), new_offset);
                        self.current_offset = new_offset;
                        Operand::Stack(new_offset)
                    }
                }
            }
            other => other.clone(),
        }
    }

    fn replace_pseudos_in_instruction(
        &mut self,
        instruction: &Instruction,
        symbols: &SymbolTable,
    ) -> Instruction {
        match instruction {
            Instruction::Mov(src, dst) => {
                let new_src = self.replace_operand(src, symbols);
                let new_dst = self.replace_operand(dst, symbols);
                Instruction::Mov(new_src, new_dst)
            }
            Instruction::Unary(op, dst) => {
                let new_dst = self.replace_operand(dst, symbols);
                Instruction::Unary(op.clone(), new_dst)
            }
            Instruction::Binary { op, src, dst } => {
                let new_src = self.replace_operand(src, symbols);
                let new_dst = self.replace_operand(dst, symbols);
                Instruction::Binary {
                    op: op.clone(),
                    src: new_src,
                    dst: new_dst,
                }
            }
            Instruction::Cmp(op1, op2) => {
                let new_op1 = self.replace_operand(op1, symbols);
                let new_op2 = self.replace_operand(op2, symbols);
                Instruction::Cmp(new_op1, new_op2)
            }
            Instruction::Idiv(op) => {
                let new_op = self.replace_operand(op, symbols);
                Instruction::Idiv(new_op)
            }
            Instruction::SetCC(code, op) => {
                let new_op = self.replace_operand(op, symbols);
                Instruction::SetCC(code.clone(), new_op)
            }
            Instruction::Push(op) => {
                let new_op = self.replace_operand(op, symbols);
                Instruction::Push(new_op)
            }
            Instruction::Ret
            | Instruction::Cdq
            | Instruction::Label(_)
            | Instruction::JmpCC(_, _)
            | Instruction::Jmp(_)
            | Instruction::DeallocateStack(_)
            | Instruction::Call(_)
            | Instruction::AllocateStack(_) => instruction.clone(),
        }
    }
}

fn replace_pseudos_in_tl(top_level: TopLevel, symbol_table: &mut SymbolTable) -> TopLevel {
    match top_level {
        TopLevel::Function {
            name,
            global,
            instructions,
        } => {
            let mut state = ReplacementState::new();
            let fixed_instructions = instructions
                .into_iter()
                .map(|instr| state.replace_pseudos_in_instruction(&instr, &symbol_table))
                .collect();
            symbol_table.set_bytes_required(&name, state.current_offset);
            TopLevel::Function {
                name: name.to_string(),
                global,
                instructions: fixed_instructions,
            }
        }
        TopLevel::StaticVariable { .. } => top_level,
    }
}

pub fn replace_pseudos(program: Program, symbol_table: &mut SymbolTable) -> Program {
    let fixed_defs = program
        .top_levels
        .into_iter()
        .map(|tl| replace_pseudos_in_tl(tl, symbol_table))
        .collect::<Vec<_>>();
    Program {
        top_levels: fixed_defs,
    }
}
