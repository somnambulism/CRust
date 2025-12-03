use std::collections::HashMap;

use crate::library::{
    ast::{
        block_items::{
            Block, BlockItem, Declaration, ForInit, FunctionDeclaration, Program, Statement,
            VariableDeclaration,
        },
        storage_class::StorageClass,
        untyped_exp::{Exp, Initializer},
    },
    util::unique_ids::MAKE_NAMED_TEMPORARY,
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
                // recursively process lhs and rhs
                Exp::Assignment(
                    self.resolve_exp(left).into(),
                    self.resolve_exp(right).into(),
                )
            }
            Exp::CompoundAssign(op, left, right) => {
                // recursively process lhs and rhs
                Exp::CompoundAssign(
                    op.clone(),
                    Box::new(self.resolve_exp(left)),
                    Box::new(self.resolve_exp(right)),
                )
            }
            Exp::PostfixIncr(e) => Exp::PostfixIncr(Box::new(self.resolve_exp(e))),
            Exp::PostfixDecr(e) => Exp::PostfixDecr(Box::new(self.resolve_exp(e))),
            Exp::Var(v) => {
                // rename var from map
                self.id_map.get(v.as_str()).map_or_else(
                    || panic!("Undeclared variable: {}", v),
                    |v| Exp::Var(v.unique_name.clone()),
                )
            }
            // recursively process operands of other expressions
            Exp::Cast { target_type, e } => Exp::Cast {
                target_type: target_type.clone(),
                e: self.resolve_exp(e).into(),
            },
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
            Exp::Dereference(inner) => Exp::Dereference(self.resolve_exp(inner).into()),
            Exp::AddrOf(inner) => Exp::AddrOf(self.resolve_exp(inner).into()),
            Exp::Subscript { ptr, index } => Exp::Subscript {
                ptr: self.resolve_exp(ptr).into(),
                index: self.resolve_exp(index).into(),
            },
            // Nothing to do for constant
            Exp::Constant(c) => Exp::Constant(c.clone()),
            Exp::Init(_) => panic!("Internal error: Unexpected initializer"),
        }
    }

    fn resolve_optional_exp(&mut self, exp: Option<Exp>) -> Option<Exp> {
        exp.map(|e| self.resolve_exp(&e))
    }

    fn resolve_local_var_helper(
        &mut self,
        name: &str,
        storage_class: &Option<StorageClass>,
    ) -> String {
        match self.id_map.get(name) {
            Some(VarEntry {
                from_current_scope: true,
                has_linkage,
                ..
            }) => {
                if !(*has_linkage && storage_class == &Some(StorageClass::Extern)) {
                    // variable is present in the map and was declared in the current block
                    panic!("Duplicate variable declaration");
                }
            }
            _ => (),
        }
        let entry = if storage_class == &Some(StorageClass::Extern) {
            VarEntry {
                unique_name: name.to_string(),
                from_current_scope: true,
                has_linkage: true,
            }
        } else {
            // variable isn't in the map, or was defined in an outer scope;
            // generate a unique name and add it to the map
            let unique_name = MAKE_NAMED_TEMPORARY(name);
            VarEntry {
                unique_name: unique_name.clone(),
                from_current_scope: true,
                has_linkage: false,
            }
        };

        self.id_map.insert(name.to_string(), entry.clone());
        entry.unique_name
    }

    fn resolve_initializer(&self, init: &Initializer) -> Initializer {
        match init {
            Initializer::SingleInit(e) => Initializer::SingleInit(self.resolve_exp(e)),
            Initializer::CompoundInit(inits) => Initializer::CompoundInit(
                inits
                    .iter()
                    .map(|init| self.resolve_initializer(init).into())
                    .collect(),
            ),
        }
    }

    fn resolve_local_var_declaration(
        &mut self,
        VariableDeclaration {
            name,
            var_type,
            init,
            storage_class,
        }: VariableDeclaration<Exp>,
    ) -> VariableDeclaration<Exp> {
        let unique_name = self.resolve_local_var_helper(&name, &storage_class);

        let init = match init {
            Some(Exp::Init(i)) => Some(*i),
            Some(_) => panic!("Internal error: incorrect initializer"),
            None => None,
        };
        let resolved_init = init.map(|init| Exp::Init(self.resolve_initializer(&init).into()));

        VariableDeclaration {
            name: unique_name,
            var_type,
            init: resolved_init,
            storage_class,
        }
    }

    fn resolve_for_init(&mut self, init: ForInit<Exp>) -> ForInit<Exp> {
        match init {
            ForInit::InitExp(e) => ForInit::InitExp(self.resolve_optional_exp(e)),
            ForInit::InitDecl(d) => {
                let resolved_d = self.resolve_local_var_declaration(d);
                ForInit::InitDecl(resolved_d)
            }
        }
    }

    fn resolve_statement(&mut self, statement: Statement<Exp>) -> Statement<Exp> {
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
                control,
                body,
                cases,
                id,
            } => Statement::Switch {
                control: self.resolve_exp(&control),
                body: self.resolve_statement(*body).into(),
                cases,
                id,
            },
            Statement::Case(v, stmt, id) => {
                Statement::Case(v, self.resolve_statement(*stmt).into(), id)
            }
            Statement::Default(stmt, id) => {
                Statement::Default(self.resolve_statement(*stmt).into(), id)
            }
            Statement::Null | Statement::Break(_) | Statement::Continue(_) => statement,
            Statement::LabelledStatement(lbl, stmt) => {
                Statement::LabelledStatement(lbl, self.resolve_statement(*stmt).into())
            }
            _ => statement,
        }
    }

    fn resolve_block_item(&mut self, item: BlockItem<Exp>) -> BlockItem<Exp> {
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

    fn resolve_block(&mut self, Block(items): Block<Exp>) -> Block<Exp> {
        let resolved_items = items
            .into_iter()
            .map(|item| self.resolve_block_item(item))
            .collect();
        Block(resolved_items)
    }

    fn resolve_local_declaration(&mut self, d: Declaration<Exp>) -> Declaration<Exp> {
        match d {
            Declaration::VarDecl(vd) => {
                let resolved_vd = self.resolve_local_var_declaration(vd);
                Declaration::VarDecl(resolved_vd)
            }
            Declaration::FunDecl(FunctionDeclaration { body: Some(_), .. }) => {
                panic!("nested function declarations are not allowed")
            }
            Declaration::FunDecl(FunctionDeclaration {
                storage_class: Some(StorageClass::Static),
                ..
            }) => {
                panic!("static keyword not allowed on local function declarations")
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
            .map(|param| self.resolve_local_var_helper(&param, &None))
            .collect()
    }

    fn resolve_function_declaration(
        &mut self,
        func: FunctionDeclaration<Exp>,
    ) -> FunctionDeclaration<Exp> {
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

    fn resolve_file_scope_variable_declaration(
        &mut self,
        vd: VariableDeclaration<Exp>,
    ) -> VariableDeclaration<Exp> {
        let name = vd.name.clone();
        self.id_map.insert(
            name.clone(),
            VarEntry {
                unique_name: name,
                from_current_scope: true,
                has_linkage: true,
            },
        );
        vd
    }

    fn resolve_global_declaration(&mut self, d: Declaration<Exp>) -> Declaration<Exp> {
        match d {
            Declaration::FunDecl(fd) => {
                let fd = self.resolve_function_declaration(fd);
                Declaration::FunDecl(fd)
            }
            Declaration::VarDecl(vd) => {
                let resolved_vd = self.resolve_file_scope_variable_declaration(vd);
                Declaration::VarDecl(resolved_vd)
            }
        }
    }

    pub fn resolve(&mut self, Program(decls): Program<Exp>) -> Program<Exp> {
        Program(
            decls
                .into_iter()
                .map(|decl| self.resolve_global_declaration(decl))
                .collect(),
        )
    }
}
