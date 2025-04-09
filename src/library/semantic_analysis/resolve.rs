use core::panic;
use std::collections::HashMap;

use crate::library::{
    ast::{BlockItem, Declaration, Exp, FunctionDefinition, Program, Statement},
    unique_ids::make_named_temporary,
};

pub struct Resolver {
    var_map: HashMap<String, String>,
}

impl Resolver {
    pub fn new() -> Self {
        Resolver {
            var_map: HashMap::new(),
        }
    }

    fn resolve_exp(&self, exp: Exp) -> Exp {
        match exp {
            Exp::Assignment(left, right) => {
                // validate that lhs is an lvalue
                if let Exp::Var(_) = *left {
                    // recursively process lhs and rhs
                    return Exp::Assignment(
                        self.resolve_exp(*left).into(),
                        self.resolve_exp(*right).into(),
                    );
                } else {
                    panic!(
                        "Expected expression on left-hand side of assignment statement, found {:?}",
                        left
                    )
                }
            }
            Exp::Var(v) => {
                // rename var from map
                self.var_map.get(v.as_str()).map_or_else(
                    || panic!("Undeclared variable: {}", v),
                    |name| Exp::Var(name.clone()),
                )
            }
            // recursively process operands for unary and binary
            Exp::Unary(op, e) => Exp::Unary(op, Box::new(self.resolve_exp(*e))),
            Exp::Binary(op, e1, e2) => Exp::Binary(
                op,
                Box::new(self.resolve_exp(*e1)),
                Box::new(self.resolve_exp(*e2)),
            ),
            // Nothing to do for constant
            Exp::Constant(_) => exp,
        }
    }

    fn resolve_declaration(&mut self, Declaration { name, init }: Declaration) -> Declaration {
        if self.var_map.contains_key(&name) {
            panic!("Duplicate variable declaration");
        }

        // Generate a unique name and add it to the map
        let unique_name = make_named_temporary(&name);
        self.var_map.insert(name, unique_name.clone());

        // resolve the initializer if there is one
        let resolved_init = match init {
            Some(e) => Some(self.resolve_exp(e)),
            None => None,
        };

        Declaration {
            name: unique_name,
            init: resolved_init,
        }
    }

    fn resolve_statement(&self, statement: Statement) -> Statement {
        match statement {
            Statement::Return(e) => Statement::Return(self.resolve_exp(e)),
            Statement::Expression(e) => Statement::Expression(self.resolve_exp(e)),
            Statement::Null => Statement::Null,
        }
    }

    fn resolve_block_item(&mut self, item: BlockItem) -> BlockItem {
        match item {
            BlockItem::S(s) => {
                // resolving a statement does not change the variable map
                let resolved_s = self.resolve_statement(s);
                BlockItem::S(resolved_s)
            }
            BlockItem::D(d) => {
                // resolving a declaration does change the variable map
                let resolved_d = self.resolve_declaration(d);
                BlockItem::D(resolved_d)
            }
        }
    }

    fn resolve_function_def(
        &mut self,
        FunctionDefinition { name, body }: FunctionDefinition,
    ) -> FunctionDefinition {
        let resolved_body: Vec<BlockItem> = body
            .into_iter()
            .map(|item| self.resolve_block_item(item))
            .collect();

        FunctionDefinition {
            name,
            body: resolved_body,
        }
    }

    pub fn resolve(&mut self, program: Program) -> Program {
        Program {
            function: self.resolve_function_def(program.function),
        }
    }
}
