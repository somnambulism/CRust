use std::collections::HashMap;

use super::types::Type;

#[derive(Debug, Clone, PartialEq)]
pub enum InitialValue {
    Tentative,
    Initial(i64),
    NoInitializer,
}

#[derive(Debug, Clone, PartialEq)]
pub enum IdentifierAttrs {
    FunAttr {
        defined: bool,
        global: bool,
        stack_frame_size: isize,
    },
    StaticAttr {
        init: InitialValue,
        global: bool,
    },
    LocalAttr,
}

#[derive(Debug)]
pub struct Entry {
    pub t: Type,
    pub attrs: IdentifierAttrs,
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

    // Apply f to value at k in HashMap
    pub fn modify<F>(&mut self, name: &str, f: F)
    where
        F: FnOnce(Entry) -> Entry,
    {
        if let Some(entry) = self.table.remove(name) {
            self.table.insert(name.to_string(), f(entry));
        } else {
            panic!("No entry found for symbol: {}", name);
        }
    }

    pub fn add_automatic_var(&mut self, name: &str, t: Type) {
        self.table.insert(
            name.to_string(),
            Entry {
                t,
                attrs: IdentifierAttrs::LocalAttr,
            },
        );
    }

    pub fn add_static_var(&mut self, name: &str, t: Type, global: bool, init: InitialValue) {
        self.table.insert(
            name.to_string(),
            Entry {
                t,
                attrs: { IdentifierAttrs::StaticAttr { init, global } },
            },
        );
    }

    pub fn add_fun(&mut self, name: &str, t: Type, global: bool, defined: bool) {
        self.table.insert(
            name.to_string(),
            Entry {
                t,
                attrs: IdentifierAttrs::FunAttr {
                    global,
                    defined,
                    stack_frame_size: 0,
                },
            },
        );
    }

    pub fn get(&self, name: &str) -> &Entry {
        self.table
            .get(name)
            .unwrap_or_else(|| panic!("{} not found in the symbol table", name))
    }

    pub fn get_opt(&self, name: &str) -> Option<&Entry> {
        self.table.get(name)
    }

    pub fn is_global(&self, name: &str) -> bool {
        self.table
            .get(name)
            .map_or(false, |entry| match &entry.attrs {
                IdentifierAttrs::LocalAttr => false,
                IdentifierAttrs::StaticAttr { global, .. } => *global,
                IdentifierAttrs::FunAttr { global, .. } => *global,
            })
    }

    pub fn is_static(&self, name: &str) -> bool {
        self.table
            .get(name)
            .map_or(false, |entry| match &entry.attrs {
                IdentifierAttrs::LocalAttr => false,
                IdentifierAttrs::StaticAttr { .. } => true,
                IdentifierAttrs::FunAttr { .. } => {
                    panic!("Internal error: functions don't have storage duration")
                }
            })
    }

    pub fn bindings(&self) -> Vec<(String, &Entry)> {
        self.table
            .iter()
            .map(|(name, entry)| (name.clone(), entry))
            .collect()
    }

    pub fn is_defined(&self, name: &str) -> bool {
        self.table.contains_key(name)
    }

    pub fn set_bytes_required(&mut self, name: &str, bytes_required: isize) {
        self.modify(name, |mut entry| match &mut entry.attrs {
            IdentifierAttrs::FunAttr {
                stack_frame_size, ..
            } => {
                *stack_frame_size = bytes_required;
                entry
            }
            _ => {
                panic!("Internal error: not a function");
            }
        });
    }

    pub fn get_bytes_required(&self, name: &str) -> isize {
        match self.get(name).attrs {
            IdentifierAttrs::FunAttr {
                stack_frame_size, ..
            } => stack_frame_size,
            _ => {
                panic!("Internal error: not a function");
            }
        }
    }
}
