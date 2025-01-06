### This is PyBase: A BASIC-like language with an interpreter written completely in Python!

# PyBase Programming Language Documentation

### Execution
There are 2 ways to execute a PyBase Script

- **Through the PyBase Shell**:
    1. Open the PyBase Shell
    2. Execute the `RUN()` function with the path to the `.pyb` file
- **Using pybase.py**:
    1. Open a terminal window
    2. Run `python pybase.py {PATH_TO_FILE}` but replace `{PATH_TO_FILE}` with the path to the `.pyb` file

## Global Constants
These constants are predefined values in the PyBase language:

- **TRUE**: Boolean value representing logical true.
- **FALSE**: Boolean value representing logical false.
- **NULL**: Represents a null or undefined value.
- **PI**: The mathematical constant π (approximately 3.14159).

## Syntax and Usages

### Comments
- **Single-line comment**: Prefixed with `#`.
- **Multi-line comment**: Prefixed with `//`.

### Expressions
- **Arithmetic operations**: Supports standard arithmetic operators such as `+`, `-`, `*`, `/`, and more.
- **Logical operations**: Supports logical operators like `AND`, `OR`, `NOT`.
- **Multi-line statements**: Use `;` and `\n` to separate statements in the same line.

### Variable Declaration
- **Declaring a variable**:  
    Syntax:  
    `VAR <var_name> = <expr>`

    Example:  
    `VAR age = 30`  
    `VAR name = "Alice"`

### Conditional Statements
- **If statement**:  
    Syntax:  
    `IF <expr> THEN <expr> (ELIF <expr> THEN <expr>)? (ELSE <expr>)?`

    Example:  
    `IF age >= 18 THEN PRINT("Adult") ELSE PRINT("Minor")`

### Looping Statements
- **For loop**:  
    Syntax:  
    `FOR <var_name> = <start_value> TO <end_value> (STEP <step_value>)? THEN <expr>`

    Example:  
    `FOR i = 1 TO 5 THEN PRINT(i)`

- **While loop**:  
    Syntax:  
    `WHILE <condition> THEN <expr>`

    Example:  
    `VAR x = 0`  
    `WHILE x < 10 THEN; PRINT(x); VAR x = x + 1; END`

### Functions
- **Function definition**:  
    Syntax:  
    `FUNC <func_name> (<args>*) -> <expr>`

    Example:  
    `FUNC greet(name) -> PRINT("Hello, " + name)`

- **Function call**:  
    Syntax:  
    `<func_name>(<args>?)`

    Example:  
    `greet("Bob")`

### List Operations
- **Adding an element to a list**: Use the `+` operator.  
    Syntax:  
    `[1, 2, 3] + 4`  
    `# Result: [1, 2, 3, 4]`

- **Combining two lists**: Use the `*` operator.  
    Syntax:  
    `[1, 2, 3] * [3, 4, 5]`  
    `# Result: [1, 2, 3, 3, 4, 5]`

- **Removing an element at a given index**: Use the `-` operator.  
    Syntax:  
    `[1, 2, 3] - 1`  
    `# Result: [1, 3]`

- **Accessing an element at a given index**: Use the `/` operator.  
    Syntax:  
    `[1, 2, 3] / 0`  
    `# Result: 1 (element at index 0)`

## Built-in Functions

Here are the built-in function definitions available in PyBase:

- **PRINT(String) -> Null**: Prints output to the console.
- **PRINT_RET(String) -> String**: Prints output and returns the result.
- **INPUT() -> String**: Reads input as a string.
- **INPUT_INT() -> Number**: Reads input as an integer.
- **CLEAR() -> Null**: Clears the screen.
- **CLS() -> Null**: Clears the screen.
- **IS_NUM(Value) -> Boolean**: Checks if a variable is a number.
- **IS_STR(Value) -> Boolean**: Checks if a variable is a string.
- **IS_LIST(Value) -> Boolean**: Checks if a variable is a list.
- **IS_FUNC(Value) -> Boolean**: Checks if a variable is a function.
- **APPEND(List, Value) -> Null**: Adds an element to the end of a list.
- **POP(List) -> Value**: Removes the last element of a list and returns it.
- **EXTEND(List, List) -> Null**: Extends a list by adding all elements from another list.
- **LEN(Value) -> int**: Returns the length of a list or string.
- **MAP(Function, List) -> List**: Applies a function to all elements in a list and returns a new list with the results.
- **JOIN(List, String) -> String**: Joins all elements of a list into a single string with a specified separator.
- **RUN(String) -> Null**: Executes a given script.

## Additional Notes
- **Optional**: When `?` is used in a syntax, it means the part is optional and may or may not appear.
- **Zero or More**: When `*` is used in a syntax, it indicates that the part may appear zero or more times.
- To see an example of what you can do using `PyBase`, check out the example.pyb file!
