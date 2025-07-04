use std::collections::HashMap;

use crate::library::{
    ast::{
        Block, BlockItem, Declaration, Exp, ForInit, FunctionDeclaration, Program, Statement,
        VariableDeclaration,
    },
    util::unique_ids::make_named_temporary,
};

#[derive(Clone)]
struct VarEntry {
    unique_name: String,
    from_current_scope: bool,
    has_linkage: bool,
}

type VarMap = HashMap<String, VarEntry>;

pub struct Resolver {
    id_map: VarMap,
}

impl Resolver {
    pub fn new() -> Self {
        Resolver {
            id_map: HashMap::new(),
        }
    }

    fn copy_variable_map(&self) -> VarMap {
        self.id_map
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    VarEntry {
                        unique_name: v.unique_name.clone(),
                        from_current_scope: false,
                        has_linkage: v.has_linkage,
                    },
                )
            })
            .collect()
    }

    fn resolve_exp(&self, exp: &Exp) -> Exp {
        match exp {
            Exp::Assignment(left, right) => {
                // validate that lhs is an lvalue
                if let Exp::Var(_) = left.as_ref() {
                    // recursively process lhs and rhs
                    return Exp::Assignment(
                        self.resolve_exp(left).into(),
                        self.resolve_exp(right).into(),
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
                self.id_map.get(v.as_str()).map_or_else(
                    || panic!("Undeclared variable: {}", v),
                    |v| Exp::Var(v.unique_name.clone()),
                )
            }
            // recursively process operands for unary, binary and conditional
            Exp::Unary(op, e) => Exp::Unary(op.clone(), Box::new(self.resolve_exp(e))),
            Exp::Binary(op, e1, e2) => Exp::Binary(
                op.clone(),
                Box::new(self.resolve_exp(e1)),
                Box::new(self.resolve_exp(e2)),
            ),
            Exp::Conditional {
                condition,
                then_result,
                else_result,
            } => Exp::Conditional {
                condition: self.resolve_exp(condition).into(),
                then_result: self.resolve_exp(then_result).into(),
                else_result: self.resolve_exp(else_result).into(),
            },
            Exp::FunCall { f, args } => {
                let fn_name = self.id_map.get(f.as_str()).map_or_else(
                    || panic!("Undeclared function: {}", f),
                    |v| v.unique_name.clone(),
                );
                Exp::FunCall {
                    f: fn_name,
                    args: args.iter().map(|e| self.resolve_exp(e)).collect(),
                }
            }
            // Nothing to do for constant
            Exp::Constant(c) => Exp::Constant(*c),
            Exp::CompoundAssign(op, lhs, rhs) => {
                // validate that lhs is an lvalue
                if let Exp::Var(_) = lhs.as_ref() {
                    // recursively process lhs and rhs
                    return Exp::CompoundAssign(
                        op.clone(),
                        Box::new(self.resolve_exp(lhs)),
                        Box::new(self.resolve_exp(rhs)),
                    );
                } else {
                    panic!(
                        "Expected expression on left-hand side of compound assignment statement, found {:?}",
                        lhs
                    )
                }
            }
            Exp::PrefixIncrement(e) => Exp::PrefixIncrement(Box::new(self.resolve_exp(e))),
            Exp::PrefixDecrement(e) => Exp::PrefixDecrement(Box::new(self.resolve_exp(e))),
            Exp::PostfixIncrement(e) => Exp::PostfixIncrement(Box::new(self.resolve_exp(e))),
            Exp::PostfixDecrement(e) => Exp::PostfixDecrement(Box::new(self.resolve_exp(e))),
        }
    }

    fn resolve_optional_exp(&mut self, exp: Option<Exp>) -> Option<Exp> {
        exp.map(|e| self.resolve_exp(&e))
    }

    fn resolve_local_ver_helper(&mut self, name: &str) -> String {
        match self.id_map.get(name) {
            Some(VarEntry {
                from_current_scope: true,
                ..
            }) => {
                // variable is present in the map and was declared in the current block
                panic!("Duplicate variable declaration");
            }
            _ => {
                // variable isn't in the map, or was defined in an outer scope;
                // generate a unique name and add it to the map
                let unique_name = make_named_temporary(name);
                self.id_map.insert(
                    name.to_string(),
                    VarEntry {
                        unique_name: unique_name.clone(),
                        from_current_scope: true,
                        has_linkage: false,
                    },
                );
                unique_name
            }
        }
    }

    fn resolve_local_var_declaration(
        &mut self,
        VariableDeclaration { name, init }: VariableDeclaration,
    ) -> VariableDeclaration {
        let unique_name = self.resolve_local_ver_helper(&name);

        let resolved_init = init.map(|init| self.resolve_exp(&init));

        VariableDeclaration {
            name: unique_name,
            init: resolved_init,
        }
    }

    fn resolve_for_init(&mut self, init: ForInit) -> ForInit {
        match init {
            ForInit::InitExp(e) => ForInit::InitExp(self.resolve_optional_exp(e)),
            ForInit::InitDecl(d) => {
                let resolved_d = self.resolve_local_var_declaration(d);
                ForInit::InitDecl(resolved_d)
            }
        }
    }

    fn resolve_statement(&mut self, statement: Statement) -> Statement {
        match statement {
            Statement::Return(e) => Statement::Return(self.resolve_exp(&e)),
            Statement::Expression(e) => Statement::Expression(self.resolve_exp(&e)),
            Statement::If {
                condition,
                then_clause,
                else_clause,
            } => Statement::If {
                condition: self.resolve_exp(&condition),
                then_clause: self.resolve_statement(*then_clause).into(),
                else_clause: else_clause.map(|stmt| self.resolve_statement(*stmt).into()),
            },
            Statement::While {
                condition,
                body,
                id,
            } => Statement::While {
                condition: self.resolve_exp(&condition),
                body: Box::new(self.resolve_statement(*body)),
                id,
            },
            Statement::DoWhile {
                body,
                condition,
                id,
            } => Statement::DoWhile {
                body: Box::new(self.resolve_statement(*body)),
                condition: self.resolve_exp(&condition),
                id,
            },
            Statement::For {
                init,
                condition,
                post,
                body,
                id,
            } => {
                let saved = self.id_map.clone();
                self.id_map = self.copy_variable_map();

                let resolved_init = self.resolve_for_init(init);
                let resolved_body = self.resolve_statement(*body);

                let resolved_for = Statement::For {
                    init: resolved_init,
                    condition: self.resolve_optional_exp(condition),
                    post: self.resolve_optional_exp(post),
                    body: Box::new(resolved_body),
                    id,
                };

                self.id_map = saved;
                resolved_for
            }
            Statement::Compound(block) => {
                let saved = self.id_map.clone();
                self.id_map = self.copy_variable_map();
                let resolved = self.resolve_block(block);
                self.id_map = saved;
                Statement::Compound(resolved)
            }
            Statement::Switch {
                condition,
                body,
                cases,
                id,
            } => Statement::Switch {
                condition: self.resolve_exp(&condition),
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
                // let resolved_d = self.resolve_declaration(d);
                let resolved_d = self.resolve_local_declaration(d);
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

    fn resolve_local_declaration(&mut self, d: Declaration) -> Declaration {
        match d {
            Declaration::VarDecl(vd) => {
                let resolved_vd = self.resolve_local_var_declaration(vd);
                Declaration::VarDecl(resolved_vd)
            }
            Declaration::FunDecl(FunctionDeclaration { body: Some(_), .. }) => {
                panic!("nested function declarations are not allowed")
            }
            Declaration::FunDecl(fd) => {
                let resolved_fd = self.resolve_function_declaration(fd);
                Declaration::FunDecl(resolved_fd)
            }
        }
    }

    fn resolve_params(&mut self, params: &Vec<String>) -> Vec<String> {
        params
            .iter()
            .map(|param| self.resolve_local_ver_helper(&param))
            .collect()
    }

    fn resolve_function_declaration(&mut self, func: FunctionDeclaration) -> FunctionDeclaration {
        if let Some(VarEntry {
            from_current_scope: true,
            has_linkage: false,
            ..
        }) = self.id_map.get(&func.name)
        {
            panic!("Duplicate declaration {}", func.name);
        } else {
            let new_entry = VarEntry {
                unique_name: func.name.clone(),
                from_current_scope: true,
                has_linkage: true,
            };
            self.id_map.insert(func.name.clone(), new_entry);
            let saved = self.id_map.clone();
            self.id_map = self.copy_variable_map();
            let resolved_params = self.resolve_params(&func.params);
            let resolved_body = func.body.map(|body| self.resolve_block(body));
            self.id_map = saved;
            FunctionDeclaration {
                params: resolved_params,
                body: resolved_body,
                ..func
            }
        }
    }

    pub fn resolve(&mut self, Program(fn_decls): Program) -> Program {
        Program(
            fn_decls
                .into_iter()
                .map(|fn_decl| self.resolve_function_declaration(fn_decl))
                .collect(),
        )
    }
}
