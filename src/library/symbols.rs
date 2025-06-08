use std::collections::HashMap;

use super::types::Type;

#[derive(Debug)]
pub struct Entry {
    pub t: Type,
    pub is_defined: bool, // only used for functions
    pub stack_frame_size: isize,
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

    pub fn add_var(&mut self, name: &str, t: Type) {
        self.table.insert(
            name.to_string(),
            Entry {
                t,
                is_defined: false,
                stack_frame_size: 0,
            },
        );
    }

    pub fn add_fun(&mut self, name: &str, t: Type, is_defined: bool) {
        self.table.insert(
            name.to_string(),
            Entry {
                t,
                is_defined,
                stack_frame_size: 0,
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

    pub fn is_defined(&self, name: &str) -> bool {
        self.table.contains_key(name)
    }

    pub fn set_bytes_required(&mut self, name: &str, bytes_required: isize) {
        self.modify(name, |mut entry| {
            entry.stack_frame_size = bytes_required;
            entry
        });
    }
}
