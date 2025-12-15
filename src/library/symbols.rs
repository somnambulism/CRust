use indexmap::map;

use indexmap::IndexMap;

use crate::library::initializers::StaticInit;
use crate::library::util::unique_ids::MAKE_NAMED_TEMPORARY;

use super::types::Type;

#[derive(Debug, Clone, PartialEq)]
pub enum InitialValue {
    Tentative,
    Initial(Vec<StaticInit>),
    NoInitializer,
}

#[derive(Debug, Clone, PartialEq)]
pub enum IdentifierAttrs {
    FunAttr { defined: bool, global: bool },
    StaticAttr { init: InitialValue, global: bool },
    ConstAttr(StaticInit),
    LocalAttr,
}

#[derive(Debug)]
pub struct Entry {
    pub t: Type,
    pub attrs: IdentifierAttrs,
}

#[derive(Debug)]
pub struct SymbolTable {
    table: IndexMap<String, Entry>,
}

impl SymbolTable {
    pub fn new() -> Self {
        SymbolTable {
            table: IndexMap::with_capacity(20),
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
                attrs: IdentifierAttrs::FunAttr { global, defined },
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

    pub fn get_type(&self, name: &str) -> Option<Type> {
        let e = self.table.get(name);
        if let Some(entry) = e {
            Some(entry.t.clone())
        } else {
            None
        }
    }

    pub fn add_string(&mut self, s: &str) -> String {
        let str_id = MAKE_NAMED_TEMPORARY("string");
        let t = Type::Array {
            elem_type: Box::new(Type::Char),
            size: s.len() as i64 + 1,
        };
        self.table.insert(
            str_id.clone(),
            Entry {
                t,
                attrs: IdentifierAttrs::ConstAttr(StaticInit::StringInit(s.to_string(), true)),
            },
        );
        str_id
    }

    pub fn is_global(&self, name: &str) -> bool {
        self.table
            .get(name)
            .map_or(false, |entry| match &entry.attrs {
                IdentifierAttrs::LocalAttr | IdentifierAttrs::ConstAttr(_) => false,
                IdentifierAttrs::StaticAttr { global, .. } => *global,
                IdentifierAttrs::FunAttr { global, .. } => *global,
            })
    }

    pub fn bindings(&self) -> Vec<(String, &Entry)> {
        self.table
            .iter()
            .map(|(name, entry)| (name.clone(), entry))
            .collect()
    }

    pub fn iter(&self) -> SymbolTableIter<'_> {
        SymbolTableIter {
            iter: self.table.iter(),
        }
    }
}

pub struct SymbolTableIter<'a> {
    iter: map::Iter<'a, String, Entry>,
}

impl<'a> Iterator for SymbolTableIter<'a> {
    type Item = (&'a String, &'a Entry);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<'a> IntoIterator for &'a SymbolTable {
    type Item = (&'a String, &'a Entry);
    type IntoIter = map::Iter<'a, String, Entry>;

    fn into_iter(self) -> Self::IntoIter {
        self.table.iter()
    }
}
