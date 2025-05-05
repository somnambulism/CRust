use std::collections::HashMap;

use crate::library::{
    ast::{Block, BlockItem, Declaration, Exp, ForInit, FunctionDefinition, Program, Statement},
    unique_ids::make_named_temporary,
};

#[derive(Clone)]
struct VarEntry {
    unique_name: String,
    from_current_block: bool,
}

type VarMap = HashMap<String, VarEntry>;

pub struct Resolver {
    var_map: VarMap,
}

impl Resolver {
    pub fn new() -> Self {
        Resolver {
            var_map: HashMap::new(),
        }
    }

    fn copy_variable_map(&self) -> VarMap {
        self.var_map
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    VarEntry {
                        unique_name: v.unique_name.clone(),
                        from_current_block: false,
                    },
                )
            })
            .collect()
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
                    |v| Exp::Var(v.unique_name.clone()),
                )
            }
            // recursively process operands for unary, binary and conditional
            Exp::Unary(op, e) => Exp::Unary(op, Box::new(self.resolve_exp(*e))),
            Exp::Binary(op, e1, e2) => Exp::Binary(
                op,
                Box::new(self.resolve_exp(*e1)),
                Box::new(self.resolve_exp(*e2)),
            ),
            Exp::Conditional {
                condition,
                then_result,
                else_result,
            } => Exp::Conditional {
                condition: self.resolve_exp(*condition).into(),
                then_result: self.resolve_exp(*then_result).into(),
                else_result: self.resolve_exp(*else_result).into(),
            },
            // Nothing to do for constant
            Exp::Constant(_) => exp,
            Exp::CompoundAssign(op, lhs, rhs) => {
                // validate that lhs is an lvalue
                if let Exp::Var(_) = *lhs {
                    // recursively process lhs and rhs
                    return Exp::CompoundAssign(
                        op,
                        Box::new(self.resolve_exp(*lhs)),
                        Box::new(self.resolve_exp(*rhs)),
                    );
                } else {
                    panic!(
                        "Expected expression on left-hand side of compound assignment statement, found {:?}",
                        lhs
                    )
                }
            }
            Exp::PrefixIncrement(e) => Exp::PrefixIncrement(Box::new(self.resolve_exp(*e))),
            Exp::PrefixDecrement(e) => Exp::PrefixDecrement(Box::new(self.resolve_exp(*e))),
            Exp::PostfixIncrement(e) => Exp::PostfixIncrement(Box::new(self.resolve_exp(*e))),
            Exp::PostfixDecrement(e) => Exp::PostfixDecrement(Box::new(self.resolve_exp(*e))),
        }
    }

    fn resolve_optional_exp(&mut self, exp: Option<Exp>) -> Option<Exp> {
        exp.map(|e| self.resolve_exp(e))
    }

    fn resolve_declaration(&mut self, Declaration { name, init }: Declaration) -> Declaration {
        match self.var_map.get(&name) {
            Some(VarEntry {
                unique_name: _,
                from_current_block: true,
            }) => {
                // variable is present in the map and was declared in the current block
                panic!("Duplicate variable declaration");
            }
            _ => {
                // variable isn't in the map, or was defined in an outer scope;
                // generate a unique name and add it to the map
                let unique_name = make_named_temporary(&name);
                self.var_map.insert(
                    name.clone(),
                    VarEntry {
                        unique_name: unique_name.clone(),
                        from_current_block: true,
                    },
                );
                let resolved_init = init.map(|e| self.resolve_exp(e));
                Declaration {
                    name: unique_name,
                    init: resolved_init,
                }
            }
        }
    }

    fn resolve_for_init(&mut self, init: ForInit) -> ForInit {
        match init {
            ForInit::InitExp(e) => ForInit::InitExp(self.resolve_optional_exp(e)),
            ForInit::InitDecl(d) => {
                let resolved_d = self.resolve_declaration(d);
                ForInit::InitDecl(resolved_d)
            }
        }
    }

    fn resolve_statement(&mut self, statement: Statement) -> Statement {
        match statement {
            Statement::Return(e) => Statement::Return(self.resolve_exp(e)),
            Statement::Expression(e) => Statement::Expression(self.resolve_exp(e)),
            Statement::If {
                condition,
                then_clause,
                else_clause,
            } => Statement::If {
                condition: self.resolve_exp(condition),
                then_clause: self.resolve_statement(*then_clause).into(),
                else_clause: else_clause.map(|stmt| self.resolve_statement(*stmt).into()),
            },
            Statement::While {
                condition,
                body,
                id,
            } => Statement::While {
                condition: self.resolve_exp(condition),
                body: Box::new(self.resolve_statement(*body)),
                id,
            },
            Statement::DoWhile {
                body,
                condition,
                id,
            } => Statement::DoWhile {
                body: Box::new(self.resolve_statement(*body)),
                condition: self.resolve_exp(condition),
                id,
            },
            Statement::For {
                init,
                condition,
                post,
                body,
                id,
            } => {
                let saved = self.var_map.clone();
                self.var_map = self.copy_variable_map();

                let resolved_init = self.resolve_for_init(init);
                let resolved_body = self.resolve_statement(*body);

                let resolved_for = Statement::For {
                    init: resolved_init,
                    condition: self.resolve_optional_exp(condition),
                    post: self.resolve_optional_exp(post),
                    body: Box::new(resolved_body),
                    id,
                };

                self.var_map = saved;
                resolved_for
            }
            Statement::Compound(block) => {
                let saved = self.var_map.clone();
                self.var_map = self.copy_variable_map();
                let resolved = self.resolve_block(block);
                self.var_map = saved;
                Statement::Compound(resolved)
            }
            Statement::Switch {
                condition,
                body,
                cases,
                id,
            } => Statement::Switch {
                condition: self.resolve_exp(condition),
                body: self.resolve_statement(*body).into(),
                cases,
                id,
            },
            Statement::Case {
                condition,
                body,
                switch_label,
            } => Statement::Case {
                condition,
                body: self.resolve_statement(*body).into(),
                switch_label,
            },
            Statement::Default { body, switch_label } => Statement::Default {
                body: self.resolve_statement(*body).into(),
                switch_label,
            },
            Statement::Null | Statement::Break(_) | Statement::Continue(_) => statement,
            Statement::Labelled { label, statement } => Statement::Labelled {
                label,
                statement: self.resolve_statement(*statement).into(),
            },
            _ => statement,
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

    fn resolve_block(&mut self, Block(items): Block) -> Block {
        let resolved_items = items
            .into_iter()
            .map(|item| self.resolve_block_item(item))
            .collect();
        Block(resolved_items)
    }

    fn resolve_function_def(
        &mut self,
        FunctionDefinition { name, body }: FunctionDefinition,
    ) -> FunctionDefinition {
        self.var_map.clear();
        let resolved_body = self.resolve_block(body);

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
