use crate::library::{
    ast::{
        Block, BlockItem, Declaration, Exp, ForInit, FunctionDeclaration, Program, Statement,
        StorageClass, VariableDeclaration,
    },
    symbols::{Entry, IdentifierAttrs, InitialValue, SymbolTable},
    types::Type,
};

pub struct TypeChecker {
    pub symbol_table: SymbolTable,
}

impl TypeChecker {
    pub fn new() -> Self {
        TypeChecker {
            symbol_table: SymbolTable::new(),
        }
    }

    fn typecheck_exp(&mut self, exp: &Exp) {
        match exp {
            Exp::FunCall { f, args } => {
                let t = &self.symbol_table.get(f).t;
                match t {
                    Type::Int => {
                        panic!("Tried to use variable {} as function name", f);
                    }
                    Type::FunType { param_count } => {
                        if args.len() != *param_count {
                            panic!("Function {} called with wrong number of arguments", f);
                        } else {
                            args.iter().for_each(|arg| self.typecheck_exp(arg));
                        }
                    }
                }
            }
            Exp::Var(v) => {
                let t = &self.symbol_table.get(v).t;
                if let Type::FunType { .. } = t {
                    panic!("Tried to use function name {} as variable", v);
                }
            }
            Exp::Unary(_, inner) => self.typecheck_exp(&inner),
            Exp::Binary(_, e1, e2) => {
                self.typecheck_exp(&e1);
                self.typecheck_exp(&e2);
            }
            Exp::PostfixDecrement(inner)
            | Exp::PostfixIncrement(inner)
            | Exp::PrefixDecrement(inner)
            | Exp::PrefixIncrement(inner) => self.typecheck_exp(&inner),
            Exp::Assignment(lhs, rhs) | Exp::CompoundAssign(_, lhs, rhs) => {
                self.typecheck_exp(&lhs);
                self.typecheck_exp(&rhs);
            }
            Exp::Conditional {
                condition,
                then_result,
                else_result,
            } => {
                self.typecheck_exp(&condition);
                self.typecheck_exp(&then_result);
                self.typecheck_exp(&else_result);
            }
            Exp::Constant(_) => (),
        }
    }

    fn typecheck_block(&mut self, b: &Block) {
        b.0.iter().for_each(|item| self.typecheck_block_item(item));
    }

    fn typecheck_block_item(&mut self, block_item: &BlockItem) {
        match block_item {
            BlockItem::S(s) => self.typecheck_statement(s),
            BlockItem::D(d) => self.typecheck_local_decl(d),
        }
    }

    fn typecheck_statement(&mut self, statement: &Statement) {
        match statement {
            Statement::Return(e) => self.typecheck_exp(e),
            Statement::Expression(e) => self.typecheck_exp(e),
            Statement::If {
                condition,
                then_clause,
                else_clause,
            } => {
                self.typecheck_exp(condition);
                self.typecheck_statement(&then_clause);
                if let Some(else_clause) = else_clause {
                    self.typecheck_statement(&else_clause);
                }
            }
            Statement::Switch {
                condition, body, ..
            } => {
                self.typecheck_exp(condition);
                self.typecheck_statement(&body);
            }
            Statement::Case { body, .. } => self.typecheck_statement(&body),
            Statement::Default { body, .. } => self.typecheck_statement(&body),
            Statement::Compound(block) => self.typecheck_block(block),
            Statement::While {
                condition, body, ..
            } => {
                self.typecheck_exp(condition);
                self.typecheck_statement(&body);
            }
            Statement::DoWhile {
                body, condition, ..
            } => {
                self.typecheck_statement(&body);
                self.typecheck_exp(condition);
            }
            Statement::For {
                init,
                condition,
                post,
                body,
                ..
            } => {
                match init {
                    ForInit::InitDecl(VariableDeclaration {
                        storage_class: Some(_),
                        ..
                    }) => {
                        panic!("Storage class not permitted on declaration in for loop header");
                    }
                    ForInit::InitDecl(d) => self.typecheck_local_var_decl(&d),
                    ForInit::InitExp(e) => {
                        if let Some(e) = e {
                            self.typecheck_exp(e);
                        }
                    }
                }
                if let Some(condition) = condition {
                    self.typecheck_exp(condition);
                }
                if let Some(post) = post {
                    self.typecheck_exp(post);
                }
                self.typecheck_statement(&body);
            }
            Statement::Labelled { statement, .. } => self.typecheck_statement(statement),
            Statement::Null | Statement::Break(_) | Statement::Continue(_) | Statement::Goto(_) => {
                ()
            }
        }
    }

    fn typecheck_local_decl(&mut self, decl: &Declaration) {
        match decl {
            Declaration::VarDecl(vd) => self.typecheck_local_var_decl(vd),
            Declaration::FunDecl(fd) => self.typecheck_fn_decl(fd),
        }
    }

    fn typecheck_local_var_decl(&mut self, var_decl: &VariableDeclaration) {
        let VariableDeclaration {
            name,
            init,
            storage_class,
        } = var_decl;
        match storage_class {
            Some(StorageClass::Extern) => {
                if init.is_some() {
                    panic!("initializer on local extern declaration");
                }
                match self.symbol_table.get_opt(name.as_str()) {
                    Some(Entry { t, .. }) => {
                        // If an external local var is already in the symbol table, don't need
                        // to add it
                        if t != &Type::Int {
                            panic!("Function redeclared as variable");
                        }
                    }
                    None => {
                        self.symbol_table.add_static_var(
                            name,
                            Type::Int,
                            true,
                            InitialValue::NoInitializer,
                        );
                    }
                }
            }
            Some(StorageClass::Static) => {
                let ini = match init {
                    Some(Exp::Constant(i)) => InitialValue::Initial(i.clone()),
                    None => InitialValue::Initial(0),
                    Some(_) => {
                        panic!("non-constant initializer for static local variable");
                    }
                };
                self.symbol_table
                    .add_static_var(name, Type::Int, false, ini);
            }
            None => {
                self.symbol_table.add_automatic_var(name, Type::Int);
                if let Some(init) = init {
                    self.typecheck_exp(init);
                }
            }
        }
    }

    fn typecheck_fn_decl(&mut self, func_decl: &FunctionDeclaration) {
        let FunctionDeclaration {
            name,
            params,
            body,
            storage_class,
        } = func_decl;

        let fun_type = Type::FunType {
            param_count: params.len(),
        };
        let has_body = body.is_some();

        let global = *storage_class != Some(StorageClass::Static);
        // helper function to reconcile current and previous declarations
        let check_against_previous = |entry: &Entry| {
            let Entry { t: prev_t, attrs } = entry;
            if prev_t != &fun_type {
                panic!("Redeclared function {} with a different type", name);
            } else {
                match attrs {
                    IdentifierAttrs::FunAttr {
                        global: prev_global,
                        defined: prev_defined,
                        ..
                    } => {
                        if *prev_defined && has_body {
                            panic!("Defined body of function {} twice", name);
                        } else if *prev_global && *storage_class == Some(StorageClass::Static) {
                            panic!("Static function declaration follows non-static");
                        } else {
                            let defined = has_body || *prev_defined;
                            (defined, *prev_global)
                        }
                    }
                    _ => {
                        panic!(
                            "Internal error: symbol has function type but not function attributes"
                        );
                    }
                }
            }
        };

        let old_decl = self.symbol_table.get_opt(name.as_str());

        let (defined, global) = if let Some(old_d) = old_decl {
            check_against_previous(old_d)
        } else {
            (has_body, global)
        };

        self.symbol_table
            .add_fun(name.as_str(), fun_type, global, defined);

        if has_body {
            for p in params {
                self.symbol_table.add_automatic_var(&p, Type::Int)
            }
        }

        if let Some(body) = body {
            self.typecheck_block(body);
        }
    }

    fn typecheck_file_scope_var_decl(&mut self, var_decl: &VariableDeclaration) {
        let VariableDeclaration {
            name,
            init,
            storage_class,
        } = var_decl;
        let current_init = match init {
            Some(Exp::Constant(c)) => InitialValue::Initial(c.clone()),
            None => {
                if *storage_class == Some(StorageClass::Extern) {
                    InitialValue::NoInitializer
                } else {
                    InitialValue::Tentative
                }
            }
            Some(_) => {
                panic!("File scope variable has non-constant initializer");
            }
        };
        let current_global = *storage_class != Some(StorageClass::Static);
        let old_decl = self.symbol_table.get_opt(name.as_str());
        let (global, init) = if let Some(old_d) = old_decl {
            let Entry { t, attrs } = old_d;
            if t != &Type::Int {
                panic!("Function redeclared as variable");
            } else {
                if let IdentifierAttrs::StaticAttr {
                    init: prev_init,
                    global: prev_global,
                } = attrs
                {
                    let global = if storage_class == &Some(StorageClass::Extern) {
                        *prev_global
                    } else if current_global == *prev_global {
                        current_global
                    } else {
                        panic!("Conflicting variable linkage");
                    };
                    let init = match (prev_init, &current_init) {
                        (InitialValue::Initial(_), InitialValue::Initial(_)) => {
                            panic!("Conflicting global variable definition");
                        }
                        (InitialValue::Initial(_), _) => prev_init.clone(),
                        (
                            InitialValue::Tentative,
                            InitialValue::Tentative | InitialValue::NoInitializer,
                        ) => InitialValue::Tentative,
                        (_, InitialValue::Initial(_)) | (InitialValue::NoInitializer, _) => {
                            current_init.clone()
                        }
                    };
                    (global, init)
                } else {
                    panic!(
                        "Internal error, file-scope variable previously declared as local variable or function"
                    );
                }
            }
        } else {
            (current_global, current_init)
        };
        self.symbol_table
            .add_static_var(name, Type::Int, global, init);
    }

    fn typecheck_global_decl(&mut self, decl: &Declaration) {
        match decl {
            Declaration::FunDecl(fd) => self.typecheck_fn_decl(fd),
            Declaration::VarDecl(vd) => self.typecheck_file_scope_var_decl(vd),
        }
    }

    pub fn typecheck(&mut self, program: &Program) {
        program
            .0
            .iter()
            .for_each(|decl| self.typecheck_global_decl(decl));
    }
}
