# Elementary C Compiler Written in Rust

Hello, drugalyok!

It is a very simple C compiler written in Rust. It still requires [gcc](https://gcc.gnu.org/) for its
operation to build an executable from assembly language code. And to preprocess
the source code into C...

**Important**: 
* I don't know the Rust programming language very well, so this code
may look like shit to a professional.
* Tested on Windows only.

## Features Implemented to Date

* Compilation of the main function containing a single operator `return <int>`. 😐

## Usage

(On Windows, it is preferable to use [Git Bash](https://git-scm.com/downloads))

1. Download and install [gcc](https://gcc.gnu.org/). (And now you have a C language compiler. I don't know why you need anything else...)
2. Download and install [Rust](https://www.rust-lang.org/).
3. Clone this repo.
    ```bash
    git clone https://github.com/somnambulism/CRust.git
    ```
4. Go to the project folder and build the project.
    ```
    cd CRust
    cargo build
    ```
5. Create a C source code file containing a single `main` function that returns an integer value:
    ```C
    int main(void) {
        return 2;
    }
    ```
6. Save this file, for example, to `sources/example.c`.
7. Compile the program!
    ```bash
    (Windows)
    ./target/debug/CRust.exe sources/example.c

    (Linux)
    ./target/debug/CRust sources/example.c
    ```
8. Run the compiled binary and enhoy your exit code:
    ```bash
    (Windows)
    $ ./sources/example.exe
    $ echo $?
    2

    (Linux)
    $ ./sources/example
    $ echo $?
    2
    ```
