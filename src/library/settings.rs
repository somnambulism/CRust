#[derive(Clone, Debug, PartialEq)]
pub enum Stage {
    Lex,
    Parse,
    Validate,
    Tacky,
    Codegen,
    Assembly,
    Obj,
    Executable,
}

#[derive(PartialEq, Debug)]
pub enum Target {
    OsX,
    Linux,
    Windows,
}

pub fn current_platform() -> Target {
    if cfg!(target_os = "macos") {
        Target::OsX
    } else if cfg!(target_os = "linux") {
        Target::Linux
    } else if cfg!(target_os = "windows") {
        Target::Windows
    } else {
        panic!("Unsupported platform");
    }
}
