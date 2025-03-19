use std::fs::File;
use std::io::{Result, Write};

use super::assembly::{FunctionDefinition, Program};
use super::{
    assembly::{Instruction, Operand},
    settings::Target,
};

pub struct CodeEmitter {
    platform: Target,
    file: File,
}

impl CodeEmitter {
    pub fn new(platform: Target, filename: &str) -> Result<Self> {
        let file = File::create(filename)?;
        Ok(Self { platform, file })
    }

    fn show_operand(&self, operand: &Operand) -> String {
        match operand {
            Operand::Register => "%eax".to_string(),
            Operand::Imm(i) => format!("${}", i),
        }
    }

    fn show_label(&self, name: &str) -> String {
        match self.platform {
            Target::OsX => format!("_{}", name),
            _ => name.to_string(),
        }
    }

    fn emit_instruction(&mut self, instruction: &Instruction) -> Result<()> {
        match instruction {
            Instruction::Mov(src, dst) => writeln!(
                self.file,
                "\tmovl {}, {}",
                self.show_operand(&src),
                self.show_operand(&dst)
            ),
            Instruction::Ret => writeln!(self.file, "\tret"),
        }
    }

    fn emit_function(&mut self, function: &FunctionDefinition) -> Result<()> {
        let label = self.show_label(&function.name);
        writeln!(self.file, "\t.globl {}", label)?;
        writeln!(self.file, "{}:", label)?;
        if self.platform == Target::Windows {
            writeln!(self.file, "\tcall __main")?;
        }
        for instr in &function.instructions {
            self.emit_instruction(instr)?;
        }
        Ok(())
    }

    fn emit_stack_note(&mut self) -> Result<()> {
        match self.platform {
            Target::Linux => writeln!(self.file, "\t.section .note.GNU-stack,\"\",@progbits"),
            Target::Windows => writeln!(self.file, "\t.section .drectve"),
            _ => Ok(()),
        }
    }

    pub fn emit(&mut self, program: &Program) -> Result<()> {
        self.emit_function(&program.function)?;
        self.emit_stack_note()?;
        Ok(())
    }
}
