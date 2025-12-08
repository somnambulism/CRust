use std::collections::HashMap;

use crate::library::assembly::AsmType;

#[derive(Debug)]
pub enum Entry {
    Fun {
        defined: bool,
        bytes_required: isize,
    },
    Obj {
        t: AsmType,
        is_static: bool,
        constant: bool,
    },
}

#[derive(Debug)]
pub struct SymbolTable {
    table: HashMap<String, Entry>,
}

impl SymbolTable {
    pub fn new() -> Self {
        SymbolTable {
            table: HashMap::with_capacity(20),
        }
    }

    pub fn add_fun(&mut self, fun_name: &str, defined: bool) {
        self.table.insert(
            fun_name.to_string(),
            Entry::Fun {
                defined,
                bytes_required: 0,
            },
        );
    }

    pub fn add_var(&mut self, var_name: &str, t: &AsmType, is_static: bool) {
        self.table.insert(
            var_name.to_string(),
            Entry::Obj {
                t: t.clone(),
                is_static,
                constant: false,
            },
        );
    }

    pub fn add_constant(&mut self, const_name: &str, t: &AsmType) {
        self.table.insert(
            const_name.to_string(),
            Entry::Obj {
                t: t.clone(),
                is_static: true,
                constant: true,
            },
        );
    }

    pub fn set_bytes_required(&mut self, fun_name: &str, bytes_required: isize) {
        if self.table.contains_key(fun_name) {
            // note: we only set bytes_required if function is defined in this
            // translation unit
            self.table.insert(
                fun_name.to_string(),
                Entry::Fun {
                    defined: true,
                    bytes_required,
                },
            );
        } else {
            panic!("Internal error: function {} is not defined", fun_name);
        }
    }

    pub fn get_bytes_required(&self, fun_name: &str) -> isize {
        match self.table.get(fun_name).unwrap() {
            Entry::Fun { bytes_required, .. } => *bytes_required,
            Entry::Obj { .. } => {
                panic!("Internal error: {} is not a function", fun_name);
            }
        }
    }

    pub fn get_size(&self, var_name: &str) -> usize {
        match self.table.get(var_name).unwrap() {
            Entry::Obj {
                t: AsmType::Longword,
                ..
            } => 4,
            Entry::Obj {
                t: AsmType::Quadword | AsmType::Double,
                ..
            } => 8,
            Entry::Obj {
                t: AsmType::ByteArray { size, .. },
                ..
            } => *size,
            Entry::Fun { .. } => {
                panic!("Internal error: {} is a function, not an object", var_name);
            }
        }
    }

    pub fn get_alignment(&self, var_name: &str) -> usize {
        match self.table.get(var_name).unwrap() {
            Entry::Obj {
                t: AsmType::Longword,
                ..
            } => 4,
            Entry::Obj {
                t: AsmType::Quadword | AsmType::Double,
                ..
            } => 8,
            Entry::Obj {
                t: AsmType::ByteArray { alignment, .. },
                ..
            } => *alignment,
            Entry::Fun { .. } => {
                panic!("Internal error: {} is a function, not an object", var_name);
            }
        }
    }

    pub fn is_defined(&self, fun_name: &str) -> bool {
        match self.table.get(fun_name).unwrap() {
            Entry::Fun { defined, .. } => *defined,
            _ => panic!("Internal error: {} is not a function", fun_name),
        }
    }

    pub fn is_static(&self, var_name: &str) -> bool {
        match self.table.get(var_name).unwrap() {
            Entry::Obj { is_static, .. } => *is_static,
            _ => {
                panic!("Internal error: functions don't have storage duration");
            }
        }
    }

    pub fn is_constant(&self, name: &str) -> bool {
        match self.table.get(name).unwrap() {
            Entry::Obj { constant: true, .. } => true,
            Entry::Obj { .. } => false,
            Entry::Fun { .. } => {
                panic!("Internal error: is_constant doesn't make sense for functions");
            }
        }
    }
}
