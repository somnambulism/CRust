use std::collections::HashMap;

use crate::library::{
    assembly::{Instruction, Operand, Program, Reg, TopLevel},
    backend::assembly_symbols::SymbolTable,
    util::rounding::round_away_from_zero,
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
                        Operand::Memory(Reg::BP, offset)
                    // We haven't already assigned it a stack slot;
                    // assign it and update state
                    } else {
                        let size = symbols.get_size(s);
                        let alignment = symbols.get_alignment(s);
                        let new_offset = round_away_from_zero(
                            alignment as isize,
                            self.current_offset - size as isize,
                        );
                        self.offset_map.insert(s.to_string(), new_offset);
                        self.current_offset = new_offset;
                        Operand::Memory(Reg::BP, new_offset)
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
            Instruction::Mov(t, src, dst) => {
                let new_src = self.replace_operand(src, symbols);
                let new_dst = self.replace_operand(dst, symbols);
                Instruction::Mov(t.clone(), new_src, new_dst)
            }
            Instruction::Movsx(src, dst) => {
                let new_src = self.replace_operand(src, symbols);
                let new_dst = self.replace_operand(dst, symbols);
                Instruction::Movsx(new_src, new_dst)
            }
            Instruction::MovZeroExtend(src, dst) => {
                let new_src = self.replace_operand(src, symbols);
                let new_dst = self.replace_operand(dst, symbols);
                Instruction::MovZeroExtend(new_src, new_dst)
            }
            Instruction::Lea(src, dst) => {
                let new_src = self.replace_operand(src, symbols);
                let new_dst = self.replace_operand(dst, symbols);
                Instruction::Lea(new_src, new_dst)
            }
            Instruction::Unary(t, op, dst) => {
                let new_dst = self.replace_operand(dst, symbols);
                Instruction::Unary(t.clone(), op.clone(), new_dst)
            }
            Instruction::Binary { op, t, src, dst } => {
                let new_src = self.replace_operand(src, symbols);
                let new_dst = self.replace_operand(dst, symbols);
                Instruction::Binary {
                    op: op.clone(),
                    t: t.clone(),
                    src: new_src,
                    dst: new_dst,
                }
            }
            Instruction::Cmp(t, op1, op2) => {
                let new_op1 = self.replace_operand(op1, symbols);
                let new_op2 = self.replace_operand(op2, symbols);
                Instruction::Cmp(t.clone(), new_op1, new_op2)
            }
            Instruction::Idiv(t, op) => {
                let new_op = self.replace_operand(op, symbols);
                Instruction::Idiv(t.clone(), new_op)
            }
            Instruction::Div(t, op) => {
                let new_op = self.replace_operand(op, symbols);
                Instruction::Div(t.clone(), new_op)
            }
            Instruction::SetCC(code, op) => {
                let new_op = self.replace_operand(op, symbols);
                Instruction::SetCC(code.clone(), new_op)
            }
            Instruction::Push(op) => {
                let new_op = self.replace_operand(op, symbols);
                Instruction::Push(new_op)
            }
            Instruction::Cvttsd2si(t, src, dst) => {
                let new_src = self.replace_operand(src, symbols);
                let new_dst = self.replace_operand(dst, symbols);
                let new_cvt = Instruction::Cvttsd2si(t.clone(), new_src, new_dst);
                new_cvt
            }
            Instruction::Cvtsi2sd(t, src, dst) => {
                let new_src = self.replace_operand(src, symbols);
                let new_dst = self.replace_operand(dst, symbols);
                let new_cvt = Instruction::Cvtsi2sd(t.clone(), new_src, new_dst);
                new_cvt
            }
            Instruction::Ret
            | Instruction::Cdq(_)
            | Instruction::Label(_)
            | Instruction::JmpCC(_, _)
            | Instruction::Jmp(_)
            | Instruction::Call(_) => instruction.clone(),
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
        TopLevel::StaticVariable { .. } | TopLevel::StaticConstant { .. } => top_level,
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
