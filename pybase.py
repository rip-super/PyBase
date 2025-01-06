import os
import sys

# ---------- CONSTANTS ----------
DIGITS = "0123456789"
LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
LETTERS_DIGITS = LETTERS + DIGITS

# ----------- ERRORS ------------
class Error:
    def __init__(self, start_pos, end_pos, error_name, details):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.error_name = error_name
        self.details = details
    
    def string_with_arrows(self, text, pos_start, pos_end):
        result = ''

        # Calculate indices
        idx_start = max(text.rfind('\n', 0, pos_start.idx), 0)
        idx_end = text.find('\n', idx_start + 1)
        if idx_end < 0: idx_end = len(text)
        
        # Generate each line
        line_count = pos_end.ln - pos_start.ln + 1
        for i in range(line_count):
            # Calculate line columns
            line = text[idx_start:idx_end]
            col_start = pos_start.col if i == 0 else 0
            col_end = pos_end.col if i == line_count - 1 else len(line) - 1

            # Append to result
            result += line + '\n'
            result += ' ' * col_start + '^' * (col_end - col_start)

            # Re-calculate indices
            idx_start = idx_end
            idx_end = text.find('\n', idx_start + 1)
            if idx_end < 0: idx_end = len(text)

        return result.replace('\t', '')
    
    def as_string(self):
        result = f"{self.error_name}: {self.details}"
        result += f"\nFile {self.start_pos.file_name}, line {self.start_pos.ln + 1}"
        result += "\n\n" + self.string_with_arrows(self.end_pos.file_text, self.start_pos, self.end_pos)
        return result

class IllegalCharError(Error):
    def __init__(self, start_pos, end_pos, details):
        super().__init__(start_pos, end_pos, "Illegal Character", details)

class ExpectedCharError(Error):
    def __init__(self, start_pos, end_pos, details):
        super().__init__(start_pos, end_pos, "Expected Character", details)

class InvalidSyntaxError(Error):
    def __init__(self, start_pos, end_pos, details=""):
        super().__init__(start_pos, end_pos, "Invalid Syntax", details)

class RTError(Error):
    def __init__(self, start_pos, end_pos, details, context):
        super().__init__(start_pos, end_pos, "Runtime Error", details)
        self.context = context
    
    def string_with_arrows(self, text, pos_start, pos_end):
        result = ''

        # Calculate indices
        idx_start = max(text.rfind('\n', 0, pos_start.idx), 0)
        idx_end = text.find('\n', idx_start + 1)
        if idx_end < 0: idx_end = len(text)
        
        # Generate each line
        line_count = pos_end.ln - pos_start.ln + 1
        for i in range(line_count):
            # Calculate line columns
            line = text[idx_start:idx_end]
            col_start = pos_start.col if i == 0 else 0
            col_end = pos_end.col if i == line_count - 1 else len(line) - 1

            # Append to result
            result += line + '\n'
            result += ' ' * col_start + '^' * (col_end - col_start)

            # Re-calculate indices
            idx_start = idx_end
            idx_end = text.find('\n', idx_start + 1)
            if idx_end < 0: idx_end = len(text)

        return result.replace('\t', '')
    
    def as_string(self):
        result  = self.generate_traceback()
        result += f'{self.error_name}: {self.details}'
        result += '\n\n' + self.string_with_arrows(self.start_pos.file_text, self.start_pos, self.end_pos)
        return result

    def generate_traceback(self):
        result = ''
        pos = self.start_pos
        ctx = self.context

        while ctx:
            result = f'  File {pos.file_name}, line {str(pos.ln + 1)}, in {ctx.display_name}\n' + result
            pos = ctx.parent_entry_pos
            ctx = ctx.parent

        return 'Traceback (most recent call last):\n' + result

# ---------- POSITION -----------
class Position:
    def __init__(self, idx, ln, col, file_name, file_text):
        self.idx = idx
        self.ln = ln
        self.col = col
        self.file_name = file_name
        self.file_text = file_text

    def advance(self, curr_char=None):
        self.idx += 1
        self.col += 1

        if curr_char == "\n":
            self.ln += 1
            self.col = 0

        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.file_name, self.file_text)

# ----------- TOKENS ------------
# TT = Token Type
TT_INT         = "INT"
TT_FLOAT       = "FLOAT"
TT_STRING      = "STRING"
TT_IDENTIFIER  = "IDENTIFIER"
TT_KEYWORD     = "KEYWORD"
TT_PLUS        = "PLUS"
TT_MINUS       = "MINUS"
TT_MUL         = "MUL"
TT_DIV         = "DIV"
TT_POW         = "POW"
TT_EQ          = "EQ"
TT_LPAREN      = "LPAREN"
TT_RPAREN      = "RPAREN"
TT_LSQUARE     = "LSQUARE"
TT_RSQUARE     = "RSQUARE"
TT_EE          = "EE"
TT_NE          = "NE"
TT_LT          = "LT"
TT_GT          = "GT"
TT_LTE         = "LTE"
TT_GTE         = "GTE"
TT_COMMA       = "COMMA"
TT_ARROW       = "ARROW"
TT_NEWLINE     = "NEWLINE"
TT_EOF         = "EOF"

KEYWORDS = [
    "VAR",
    "AND",
    "OR",
    "NOT",
    "IF",
    "THEN",
    "ELIF",
    "ELSE",
    "FOR",
    "TO",
    "STEP",
    "WHILE",
    "FUNC",
    "END",
    "RETURN",
    "CONTINUE",
    "BREAK"
]

class Token:
    def __init__(self, type_, value=None, start_pos=None, end_pos=None):
        self.type = type_
        self.value = value

        if start_pos:
            self.start_pos = start_pos.copy()
            self.end_pos = start_pos.copy()
            self.end_pos.advance()
        
        if end_pos:
            self.end_pos = end_pos.copy()
    
    def matches(self, type_, value):
        return self.type == type_ and self.value == value

    def __repr__(self):
        if self.value: return f"{self.type}: {self.value}"
        return f"{self.type}"

# ----------- LEXER ------------
class Lexer:
    def __init__(self, file_name, text):
        self.file_name = file_name
        self.text = text
        self.pos = Position(-1, 0, -1, file_name, text)
        self.curr_char = None
        self.advance()
    
    def advance(self):
        self.pos.advance(self.curr_char)
        self.curr_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None
    
    def peek(self):
        if self.pos.idx + 1 < len(self.text):
            return self.text[self.pos.idx + 1]
        else:
            return None

    def create_tokens(self):
        tokens = []

        while self.curr_char != None:
            if self.curr_char in " \t": # space or tab
                self.advance()
            elif self.curr_char == "#":  # comment
                self.advance()
                while self.curr_char not in ";\n" and self.curr_char != None:  # skip until newline
                    self.advance()
            elif self.curr_char == "/" and self.peek() == "/":  # multi-line comment start
                self.advance()
                self.advance()  # skip over the second '/'
                while self.curr_char != "/" or self.peek() != "/" and self.curr_char != None:
                    self.advance()
                self.advance()  # skip over the second '/'
                self.advance()  # skip over the last character after the comment
            else:
                match self.curr_char:
                    case self.curr_char if self.curr_char in DIGITS:
                        tokens.append(self.create_number())
                        continue
                    case self.curr_char if self.curr_char in LETTERS:
                        tokens.append(self.create_identifier())
                        continue
                    case '"':
                        token, error = self.create_string()
                        if error: return [], error
                        tokens.append(token)
                        continue
                    case self.curr_char if self.curr_char in ";\n":
                        tokens.append(Token(TT_NEWLINE, start_pos=self.pos.copy()))
                        self.advance()
                        continue
                    case "+":
                        tokens.append(Token(TT_PLUS, start_pos=self.pos))
                    case "-":
                        tokens.append(self.create_minus_or_arrow())
                    case "*":
                        tokens.append(Token(TT_MUL, start_pos=self.pos))
                    case "/":
                        tokens.append(Token(TT_DIV, start_pos=self.pos))
                    case "(":
                        tokens.append(Token(TT_LPAREN, start_pos=self.pos))
                    case ")":
                        tokens.append(Token(TT_RPAREN, start_pos=self.pos))
                    case "[":
                        tokens.append(Token(TT_LSQUARE, start_pos=self.pos))
                    case "]":
                        tokens.append(Token(TT_RSQUARE, start_pos=self.pos))
                    case "^":
                        tokens.append(Token(TT_POW, start_pos=self.pos))
                    case "=":
                        tokens.append(self.create_equals())
                        continue
                    case "!":
                        token, error = self.create_not_equals()
                        if error: return [], error
                        tokens.append(token)
                        continue
                    case "<":
                        tokens.append(self.create_less_than())
                        continue
                    case ">":
                        tokens.append(self.create_greater_than())
                        continue
                    case ",":
                        tokens.append(Token(TT_COMMA, start_pos=self.pos))
                    
                    case _:
                        start_pos = self.pos.copy()
                        char = self.curr_char
                        self.advance()
                        return [], IllegalCharError(start_pos, self.pos, "'" + char + "'")
                self.advance()
        
        tokens.append(Token(TT_EOF, start_pos=self.pos))
        #print([f"{token.type}: {token.value}" for token in tokens])
        return tokens, None
    
    def create_number(self):
        num_str = ""
        is_float = False
        start_pos = self.pos.copy()

        while self.curr_char != None and self.curr_char in DIGITS + ".":
            if self.curr_char == ".":
                if is_float: break
                is_float = True
                num_str += "."
            else:
                num_str += self.curr_char
            
            self.advance()

        if not is_float:
            return Token(TT_INT, int(num_str), start_pos, self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), start_pos, self.pos)

    def create_string(self):
        string = ""
        start_pos = self.pos.copy()
        escape_char = False
        self.advance()

        escape_chars = {
            "n": "\n",
            "t": "\t"
        }

        while self.curr_char is not None and (self.curr_char != '"' or escape_char):
            if escape_char:
                string += escape_chars.get(self.curr_char, self.curr_char)
                escape_char = False
            else:
                if self.curr_char == "\\":
                    escape_char = True
                else:
                    string += self.curr_char
            
            self.advance()
        
        if self.curr_char != '"':
            return None, ExpectedCharError(start_pos, self.pos, 'Expected closing " for string')
        
        self.advance()
        return Token(TT_STRING, string, start_pos, self.pos), None
    
    def create_identifier(self):
        id_str = ""
        start_pos = self.pos.copy()

        while self.curr_char != None and self.curr_char in LETTERS_DIGITS + "_":
            id_str += self.curr_char
            self.advance()
        
        tok_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFIER
        return Token(tok_type, id_str, start_pos, self.pos)

    def create_not_equals(self):
        start_pos = self.pos.copy()
        self.advance()

        if self.curr_char == "=":
            self.advance()
            return Token(TT_NE, start_pos=start_pos, end_pos=self.pos), None
        
        self.advance()
        return None, ExpectedCharError(start_pos, self.pos, "'=' (after '!')")
    
    def create_equals(self):
        tok_type = TT_EQ
        start_pos = self.pos.copy()
        self.advance()

        if self.curr_char == "=":
            self.advance()
            tok_type = TT_EE
        
        return Token(tok_type, start_pos=start_pos, end_pos=self.pos)
    
    def create_less_than(self):
        tok_type = TT_LT
        start_pos = self.pos.copy()
        self.advance()

        if self.curr_char == "=":
            self.advance()
            tok_type = TT_LTE
        
        return Token(tok_type, start_pos=start_pos, end_pos=self.pos)

    def create_greater_than(self):
        tok_type = TT_GT
        start_pos = self.pos.copy()
        self.advance()

        if self.curr_char == "=":
            self.advance()
            tok_type = TT_GTE
        
        return Token(tok_type, start_pos=start_pos, end_pos=self.pos)

    def create_minus_or_arrow(self):
        tok_type = TT_MINUS
        start_pos = self.pos.copy()
        self.advance()

        if self.curr_char == ">":
            self.advance()
            tok_type = TT_ARROW
        
        return Token(tok_type, start_pos=start_pos, end_pos=self.pos)

# ----------- NODES ------------
class NumberNode:
    def __init__(self, token):
        self.token = token
        self.start_pos = self.token.start_pos
        self.end_pos = self.token.end_pos
    
    def __repr__(self):
        return f"{self.token}"

class StringNode:
    def __init__(self, token):
        self.token = token
        self.start_pos = self.token.start_pos
        self.end_pos = self.token.end_pos
    
    def __repr__(self):
        return f"{self.token}"

class ListNode:
    def __init__(self, element_nodes, start_pos, end_pos):
        self.element_nodes = element_nodes

        self.start_pos = start_pos
        self.end_pos = end_pos

class VarAccessNode:
    def __init__(self, var_name_tok):
        self.var_name_tok = var_name_tok
        self.start_pos = self.var_name_tok.start_pos
        self.end_pos = self.var_name_tok.end_pos

class VarAssignNode:
    def __init__(self, var_name_tok, value_node):
        self.var_name_tok = var_name_tok
        self.value_node = value_node

        self.start_pos = self.var_name_tok.start_pos
        self.end_pos = self.value_node.end_pos

class BinaryOperatorNode:
    def __init__(self, left_node, operator_token, right_node):
        self.left_node = left_node
        self.operator_token = operator_token
        self.right_node = right_node

        self.start_pos = self.left_node.start_pos
        self.end_pos = self.right_node.end_pos

    def __repr__(self):
        return f"({self.left_node}, {self.operator_token}, {self.right_node})"

class UnaryOperatorNode:
    def __init__(self, operator_token, node):
        self.operator_token = operator_token
        self.node = node

        self.start_pos = self.operator_token.start_pos
        self.end_pos = node.end_pos

    def __repr__(self):
        return f"({self.operator_token}, {self.node})"

class IfNode:
    def __init__(self, cases, else_case):
        self.cases = cases
        self.else_case = else_case

        self.start_pos = self.cases[0][0].start_pos
        self.end_pos = (self.else_case or self.cases[-1])[0].end_pos

class ForNode:
    def __init__(self, var_name_tok, start_value_node, end_value_node, step_value_node, body_node, should_return_null):
        self.var_name_tok = var_name_tok
        self.start_value_node = start_value_node
        self.end_value_node = end_value_node
        self.step_value_node = step_value_node
        self.body_node = body_node
        self.should_return_null = should_return_null

        self.start_pos = self.var_name_tok.start_pos
        self.end_pos = self.body_node.end_pos

class WhileNode:
    def __init__(self, condition_node, body_node, should_return_null):
        self.condition_node = condition_node
        self.body_node = body_node
        self.should_return_null = should_return_null

        self.start_pos = self.condition_node.start_pos
        self.end_pos = self.body_node.end_pos

class FuncDefNode:
    def __init__(self, var_name_tok, arg_name_toks, body_node, should_auto_return):
        self.var_name_tok = var_name_tok
        self.arg_name_toks = arg_name_toks
        self.body_node = body_node
        self.should_auto_return = should_auto_return

        if self.var_name_tok:
            self.start_pos = self.var_name_tok.start_pos
        elif len(self.arg_name_toks) > 0:
            self.start_pos = self.arg_name_toks[0].start_pos
        else:
            self.start_pos = self.body_node.start_pos
        
        self.end_pos = self.body_node.end_pos

class FuncCallNode:
    def __init__(self, node_to_call, arg_nodes):
        self.node_to_call = node_to_call
        self.arg_nodes = arg_nodes

        self.start_pos = self.node_to_call.start_pos

        if len(self.arg_nodes) > 0:
            self.end_pos = self.arg_nodes[len(self.arg_nodes) - 1].end_pos
        else:
            self.end_pos = self.node_to_call.end_pos

class ReturnNode:
    def __init__(self, node_to_return, start_pos, end_pos):
        self.node_to_return = node_to_return
        self.start_pos = start_pos
        self.end_pos = end_pos

class ContinueNode:
    def __init__(self, start_pos, end_pos):
        self.start_pos = start_pos
        self.end_pos = end_pos

class BreakNode:
    def __init__(self, start_pos, end_pos):
        self.start_pos = start_pos
        self.end_pos = end_pos

# --------- PARSE RESULT--------
class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.advance_count = 0
        self.to_reverse_count = 0

    def register_advancement(self):
        self.advance_count += 1

    def register(self, result):
        self.advance_count += result.advance_count
        if result.error: self.error = result.error
        return result.node
    
    def try_register(self, result):
        if result.error:
            self.to_reverse_count = 0
            return None
        return self.register(result)
    
    def success(self, node):
        self.node = node
        return self
    
    def failure(self, error):
        if not self.error or self.advance_count == 0:
            self.error = error
        return self

# ----------- PARSER -----------
class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.token_idx = -1
        self.advance()
    
    def advance(self):
        self.token_idx += 1
        self.update_curr_tok()
        return self.curr_tok

    def reverse(self, amount=1):
        self.token_idx -= amount
        self.update_curr_tok()
        return self.curr_tok
    
    def update_curr_tok(self):
        if self.token_idx >= 0 and self.token_idx < len(self.tokens):
            self.curr_tok = self.tokens[self.token_idx]

    def parse(self):
        res = self.statements()
        if not res.error and self.curr_tok.type != TT_EOF:
            return res.failure(InvalidSyntaxError(
                self.curr_tok.start_pos, self.curr_tok.end_pos,
                "Expected '+', '-', '*', '/', '^', '==', '!=', '<', '>', <=', '>=', 'AND' or 'OR'"
        	))
        return res

    def statements(self):
        res = ParseResult()
        statements = []
        start_pos = self.curr_tok.start_pos.copy()

        while self.curr_tok.type == TT_NEWLINE:
            res.register_advancement()
            self.advance()
        
        statement = res.register(self.statement())
        if res.error: return res
        statements.append(statement)

        more_statements = True

        while True:
            newline_count = 0
            while self.curr_tok.type == TT_NEWLINE:
                res.register_advancement()
                self.advance()
                newline_count += 1
            if newline_count == 0:
                more_statements = False

            if not more_statements: break
            statement = res.try_register(self.statement())
            if not statement:
                self.reverse(res.to_reverse_count)
                more_statements = False
                continue
            statements.append(statement)
        
        return res.success(ListNode(
            statements,
            start_pos,
            self.curr_tok.end_pos.copy()
        ))

    def statement(self):
        res = ParseResult()
        start_pos = self.curr_tok.start_pos.copy()

        if self.curr_tok.matches(TT_KEYWORD, "RETURN"):
            res.register_advancement()
            self.advance()

            expr = res.try_register(self.expr())
            if not expr:
                self.reverse(res.to_reverse_count)
            return res.success(ReturnNode(expr, start_pos, self.curr_tok.start_pos.copy()))
        
        if self.curr_tok.matches(TT_KEYWORD, "CONTINUE"):
            res.register_advancement()
            self.advance()
            return res.success(ContinueNode(start_pos, self.curr_tok.start_pos.copy()))

        if self.curr_tok.matches(TT_KEYWORD, "BREAK"):
            res.register_advancement()
            self.advance()
            return res.success(BreakNode(start_pos, self.curr_tok.start_pos.copy()))

        expr = res.register(self.expr())
        if res.error: 
            return res.failure(InvalidSyntaxError(
                self.curr_tok.start_pos, self.curr_tok.end_pos, 
                "Expected 'RETURN', 'CONTINUE', 'BREAK', 'VAR', 'NOT', 'IF', 'FOR', 'WHILE', 'FUNC', int, float, identifier, '+', '-', '[' or '('"
            ))
        
        return res.success(expr)

    def expr(self):
        res = ParseResult()
        if self.curr_tok.matches(TT_KEYWORD, "VAR"):
            res.register_advancement()
            self.advance()
            if self.curr_tok.type != TT_IDENTIFIER:
                return res.failure(InvalidSyntaxError(self.curr_tok.start_pos, self.curr_tok.end_pos, "Expected identifier"))

            var_name = self.curr_tok
            res.register_advancement()
            self.advance()

            if self.curr_tok.type != TT_EQ:
                return res.failure(InvalidSyntaxError(self.curr_tok.start_pos, self.curr_tok.end_pos, "Expected '='"))

            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            return res.success(VarAssignNode(var_name, expr))

        node = res.register(self.binary_operation(self.comp_expr, ((TT_KEYWORD, "AND"), (TT_KEYWORD, "OR"))))

        if res.error:
            return res.failure(InvalidSyntaxError(
                self.curr_tok.start_pos, self.curr_tok.end_pos, 
                "Expected 'VAR', 'NOT', 'IF', 'FOR', 'WHILE', 'FUNC', int, float, identifier, '+', '-', '[' or '('"
            ))

        return res.success(node)

    def comp_expr(self):
        res = ParseResult()

        if self.curr_tok.matches(TT_KEYWORD, "NOT"):
            op_tok = self.curr_tok
            res.register_advancement()
            self.advance()

            node = res.register(self.comp_expr())
            if res.error: return res
            return res.success(UnaryOperatorNode(op_tok, node))
        
        node = res.register(self.binary_operation(self.arith_expr, (TT_EE, TT_NE, TT_LT, TT_GT, TT_LTE, TT_GTE)))

        if res.error:
            return res.failure(InvalidSyntaxError(self.curr_tok.start_pos, self.curr_tok.end_pos, "Expected int, float, identifier, '+', '-', '[', '(', or 'NOT'"))
        
        return res.success(node)

    def arith_expr(self):
        return self.binary_operation(self.term, (TT_PLUS, TT_MINUS))

    def term(self):
        return self.binary_operation(self.factor, (TT_MUL, TT_DIV))

    def factor(self):
        res = ParseResult()
        tok = self.curr_tok

        if tok.type in (TT_PLUS, TT_MINUS):
            res.register_advancement()
            self.advance()
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOperatorNode(tok, factor))
    
        return self.power()
  
    def power(self):
        return self.binary_operation(self.call, (TT_POW, ), self.factor)

    def call(self):
        res = ParseResult()
        atom = res.register(self.atom())
        if res.error: return res

        if self.curr_tok.type == TT_LPAREN:
            res.register_advancement()
            self.advance()
            arg_nodes = []

            if self.curr_tok.type == TT_RPAREN:
                res.register_advancement()
                self.advance()
            else:
                arg_nodes.append(res.register(self.expr()))
                if res.error:
                    return res.failure(InvalidSyntaxError(
                        self.curr_tok.start_pos, self.curr_tok.end_pos,
                        "Expected ')', 'VAR', 'IF', 'FOR', 'WHILE', 'FUNC', int, float, identifier, '+', '-', '[', '(' or 'NOT'"
                    ))

                while self.curr_tok.type == TT_COMMA:
                    res.register_advancement()
                    self.advance()

                    arg_nodes.append(res.register(self.expr()))
                    if res.error: return res

                if self.curr_tok.type != TT_RPAREN:
                    return res.failure(InvalidSyntaxError(
                        self.curr_tok.start_pos, self.curr_tok.end_pos,
                        "Expected ',' or ')'"
                    ))

                res.register_advancement()
                self.advance()
            
            return res.success(FuncCallNode(atom, arg_nodes))
        
        return res.success(atom)

    def atom(self):
        res = ParseResult()
        tok = self.curr_tok

        if tok.type in (TT_INT, TT_FLOAT):
            res.register_advancement()
            self.advance()
            return res.success(NumberNode(tok))
        
        if tok.type == TT_STRING:
            res.register_advancement()
            self.advance()
            return res.success(StringNode(tok))
        
        elif tok.type == TT_IDENTIFIER:
            res.register_advancement()
            self.advance()
            return res.success(VarAccessNode(tok))

        elif tok.type == TT_LPAREN:
            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            if self.curr_tok.type == TT_RPAREN:
                res.register_advancement()
                self.advance()
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(self.curr_tok.start_pos, self.curr_tok.end_pos, "Expected ')'"))
        
        elif tok.type == TT_LSQUARE:
            list_expr = res.register(self.list_expr())
            if res.error: return res

            return res.success(list_expr)

        elif tok.matches(TT_KEYWORD, "IF"):
            if_expr = res.register(self.if_expr())
            if res.error: return res
            return res.success(if_expr)

        elif tok.matches(TT_KEYWORD, "FOR"):
            for_expr = res.register(self.for_expr())
            if res.error: return res
            return res.success(for_expr)
        
        elif tok.matches(TT_KEYWORD, "WHILE"):
            while_expr = res.register(self.while_expr())
            if res.error: return res
            return res.success(while_expr)
        
        elif tok.matches(TT_KEYWORD, "FUNC"):
            func_def = res.register(self.func_def())
            if res.error: return res
            return res.success(func_def)
        
        return res.failure(InvalidSyntaxError(self.curr_tok.start_pos, self.curr_tok.end_pos, "Expected int, float, identifier, '+', '-', '(', '[' ,'IF', 'FOR', 'WHILE', or 'FUNC'"))

    def list_expr(self):
        res = ParseResult()
        element_nodes = []
        start_pos = self.curr_tok.start_pos.copy()

        if self.curr_tok.type != TT_LSQUARE:
            return res.failure(InvalidSyntaxError(self.curr_tok.start_pos, self.curr_tok.end_pos, "Expected '['"))
        
        res.register_advancement()
        self.advance()

        if self.curr_tok.type == TT_RSQUARE:
            res.register_advancement()
            self.advance()
        else:
            element_nodes.append(res.register(self.expr()))
            if res.error:
                return res.failure(InvalidSyntaxError(
                    self.curr_tok.start_pos, self.curr_tok.end_pos,
                    "Expected ']', 'VAR', 'IF', 'FOR', 'WHILE', 'FUNC', int, float, identifier, '+', '-', '[','(' or 'NOT'"
                ))

            while self.curr_tok.type == TT_COMMA:
                res.register_advancement()
                self.advance()
                element_nodes.append(res.register(self.expr()))
                if res.error: return res

            if self.curr_tok.type != TT_RSQUARE:
                return res.failure(InvalidSyntaxError(
                    self.curr_tok.start_pos, self.curr_tok.end_pos,
                    "Expected ',' or ']'"
                ))

            res.register_advancement()
            self.advance()
        
        return res.success(ListNode(element_nodes, start_pos, self.curr_tok.end_pos.copy()))

    def if_expr_cases(self, case_keyword):
        res = ParseResult()
        cases = []
        else_case = None

        if not self.curr_tok.matches(TT_KEYWORD, case_keyword):
            return res.failure(InvalidSyntaxError(
                self.curr_tok.start_pos, self.curr_tok.end_pos, 
                f"Expected '{case_keyword}'"
            ))
        
        res.register_advancement()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res

        if not self.curr_tok.matches(TT_KEYWORD, "THEN"):
            return res.failure(InvalidSyntaxError(
                self.curr_tok.start_pos, self.curr_tok.end_pos,
                "Expected 'THEN'"
            ))
        
        res.register_advancement()
        self.advance()

        if self.curr_tok.type == TT_NEWLINE:
            res.register_advancement()
            self.advance()

            statements = res.register(self.statements())
            if res.error: return res
            cases.append((condition, statements, True))

            if self.curr_tok.matches(TT_KEYWORD, "END"):
                res.register_advancement()
                self.advance()
            else:
                all_cases = res.register(self.if_expr_b_or_c())
                if res.error: return res
                new_cases, else_cases = all_cases
                cases.extend(new_cases)
        else:
            expr = res.register(self.statement())
            if res.error: return res
            cases.append((condition, expr, False))

            all_cases = res.register(self.if_expr_b_or_c())
            if res.error: return res
            new_cases, else_case = all_cases
            cases.extend(new_cases)
        
        return res.success((cases, else_case))

    def if_expr_b(self):
        return self.if_expr_cases("ELIF")
    
    def if_expr_c(self):
        res = ParseResult()
        else_case = None

        if self.curr_tok.matches(TT_KEYWORD, "ELSE"):
            res.register_advancement()
            self.advance()

            if self.curr_tok.type == TT_NEWLINE:
                res.register_advancement()
                self.advance()

                statements = res.register(self.statements())
                if res.error: return res
                else_case = (statements, True)

                if self.curr_tok.matches(TT_KEYWORD, "END"):
                    res.register_advancement()
                    self.advance()
                else:
                    return res.failure(InvalidSyntaxError(
                        self.curr_tok.start_pos, self.curr_tok.end_pos,
                        "Expected 'END'"
                    ))
            else:
                expr = res.register(self.statement())
                if res.error: return res
                else_case = (expr, False)

        return res.success(else_case)

    def if_expr_b_or_c(self):
        res = ParseResult()
        cases, else_case = [], None

        if self.curr_tok.matches(TT_KEYWORD, "ELIF"):
            all_cases = res.register(self.if_expr_b())
            if res.error: return res
            cases, else_case = all_cases
        else:
            else_case = res.register(self.if_expr_c())
            if res.error: return res
        
        return res.success((cases, else_case))

    def if_expr(self):
        res = ParseResult()
        all_cases = res.register(self.if_expr_cases("IF"))
        if res.error: return res
        cases, else_case = all_cases
        return res.success(IfNode(cases, else_case))

    def for_expr(self):
        res = ParseResult()

        if not self.curr_tok.matches(TT_KEYWORD, "FOR"):
            return res.failure(InvalidSyntaxError(self.curr_tok.start_pos, self.curr_tok.end_pos, "Expected 'FOR'"))
        
        res.register_advancement()
        self.advance()

        if self.curr_tok.type != TT_IDENTIFIER:
            return res.failure(InvalidSyntaxError(self.curr_tok.start_pos, self.curr_tok.end_pos, "Expected identifier"))
        
        var_name = self.curr_tok
        res.register_advancement()
        self.advance()

        if self.curr_tok.type != TT_EQ:
            return res.failure(InvalidSyntaxError(self.curr_tok.start_pos, self.curr_tok.end_pos, "Expected '='"))
        
        res.register_advancement()
        self.advance()

        start_value = res.register(self.expr())
        if res.error: return res

        if not self.curr_tok.matches(TT_KEYWORD, "TO"):
            return res.failure(InvalidSyntaxError(self.curr_tok.start_pos, self.curr_tok.end_pos, "Expected 'TO'"))
        
        res.register_advancement()
        self.advance()

        end_value = res.register(self.expr())
        if res.error: return res

        if self.curr_tok.matches(TT_KEYWORD, "STEP"):
            res.register_advancement()
            self.advance()

            step_value = res.register(self.expr())
            if res.error: return res
        else:
            step_value = None
        
        if not self.curr_tok.matches(TT_KEYWORD, "THEN"):
            return res.failure(InvalidSyntaxError(self.curr_tok.start_pos, self.curr_tok.end_pos, "Expected 'THEN'"))
        
        res.register_advancement()
        self.advance()

        if self.curr_tok.type == TT_NEWLINE:
            res.register_advancement()
            self.advance()

            body = res.register(self.statements())
            if res.error: return res

            if not self.curr_tok.matches(TT_KEYWORD, "END"):
                return res.failure(InvalidSyntaxError(
                    self.curr_tok.start_pos, self.curr_tok.end_pos,
                    "Expected 'END'"
                ))
        
            res.register_advancement()
            self.advance()

            return res.success(ForNode(var_name, start_value, end_value, step_value, body, True))

        body = res.register(self.statement())
        if res.error: return res

        return res.success(ForNode(var_name, start_value, end_value, step_value, body, False))
  
    def while_expr(self):
        res = ParseResult()

        if not self.curr_tok.matches(TT_KEYWORD, "WHILE"):
            return res.failure(InvalidSyntaxError(self.curr_tok.start_pos, self.curr_tok.end_pos, "Expected 'WHILE'"))
        
        res.register_advancement()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res

        if not self.curr_tok.matches(TT_KEYWORD, "THEN"):
            return res.failure(InvalidSyntaxError(self.curr_tok.start_pos, self.curr_tok.end_pos, "Expected 'THEN'"))

        res.register_advancement()
        self.advance()

        if self.curr_tok.type == TT_NEWLINE:
            res.register_advancement()
            self.advance()

            body = res.register(self.statements())
            if res.error: return res

            if not self.curr_tok.matches(TT_KEYWORD, "END"):
                return res.failure(InvalidSyntaxError(
                    self.curr_tok.start_pos, self.curr_tok.end_pos,
                    "Expected 'END'"
                ))
        
            res.register_advancement()
            self.advance()

            return res.success(WhileNode(condition, body, True))

        body = res.register(self.statement())
        if res.error: return res

        return res.success(WhileNode(condition, body, False))

    def func_def(self):
        res = ParseResult()

        if not self.curr_tok.matches(TT_KEYWORD, "FUNC"):
            return res.failure(InvalidSyntaxError(
                self.curr_tok.start_pos, self.curr_tok.end_pos,
                "Expected 'FUNC'"
            ))

        res.register_advancement()
        self.advance()

        if self.curr_tok.type == TT_IDENTIFIER:
            var_name_tok = self.curr_tok
            res.register_advancement()
            self.advance()
            if self.curr_tok.type != TT_LPAREN:
                return res.failure(InvalidSyntaxError(
                    self.curr_tok.start_pos, self.curr_tok.end_pos,
                    "Expected '('"
                ))
        else:
            var_name_tok = None
            if self.curr_tok.type != TT_LPAREN:
                return res.failure(InvalidSyntaxError(
                    self.curr_tok.start_pos, self.curr_tok.end_pos,
                    "Expected identifier or '('"
                ))

        res.register_advancement()
        self.advance()
        arg_name_toks = []

        if self.curr_tok.type == TT_IDENTIFIER:
            arg_name_toks.append(self.curr_tok)
            res.register_advancement()
            self.advance()

            while self.curr_tok.type == TT_COMMA:
                res.register_advancement()
                self.advance()

                if self.curr_tok.type != TT_IDENTIFIER:
                    return res.failure(InvalidSyntaxError(
                        self.curr_tok.start_pos, self.curr_tok.end_pos,
                        "Expected identifier"
                    ))

                arg_name_toks.append(self.curr_tok)
                res.register_advancement()
                self.advance()

            if self.curr_tok.type != TT_RPAREN:
                return res.failure(InvalidSyntaxError(
                    self.curr_tok.start_pos, self.curr_tok.end_pos,
                    "Expected ',' or ')'"
                ))
        else:
            if self.curr_tok.type != TT_RPAREN:
                return res.failure(InvalidSyntaxError(
                    self.curr_tok.start_pos, self.curr_tok.end_pos,
                    "Expected identifier or ')'"
                ))

        res.register_advancement()
        self.advance()

        if self.curr_tok.type == TT_ARROW:
            res.register_advancement()
            self.advance()
            
            body = res.register(self.expr())
            if res.error: return res

            return res.success(FuncDefNode(
                var_name_tok,
                arg_name_toks,
                body,
                True
            ))
        
        if self.curr_tok.type != TT_NEWLINE:
            return res.failure(InvalidSyntaxError(
                self.curr_tok.start_pos, self.curr_tok.end_pos,
                "Expected '->' or NEWLINE"
            ))
        
        res.register_advancement()
        self.advance()

        body = res.register(self.statements())
        if res.error: return res

        if not self.curr_tok.matches(TT_KEYWORD, "END"):
            if self.curr_tok.type == TT_IDENTIFIER and self.curr_tok.value == "END":
                res.register_advancement()
                self.advance()
            else:
                return res.failure(InvalidSyntaxError(
                    self.curr_tok.start_pos, self.curr_tok.end_pos,
                    "Expected 'END'"
                ))
        
        res.register_advancement()
        self.advance()

        return res.success(FuncDefNode(
            var_name_tok,
            arg_name_toks,
            body,
            False
        ))

    def binary_operation(self, func_a, op_toks, func_b=None):
        if func_b == None:
            func_b = func_a

        res = ParseResult()
        left = res.register(func_a())
        if res.error: return res

        while self.curr_tok.type in op_toks or (self.curr_tok.type, self.curr_tok.value) in op_toks:
            op_tok = self.curr_tok
            res.register_advancement()
            self.advance()
            right = res.register(func_b())
            if res.error: return res
            left = BinaryOperatorNode(left, op_tok, right)

        return res.success(left)

# ------ RUNTIME RESULT --------
class RTResult:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.value = None
        self.error = None
        self.func_return_value = None
        self.loop_should_continue = False
        self.loop_should_break = False

    def register(self, res):
        if res.error: self.error = res.error
        self.func_return_value = res.func_return_value
        self.loop_should_continue = res.loop_should_continue
        self.loop_should_break = res.loop_should_break
        return res.value

    def success(self, value):
        self.reset()
        self.value = value
        return self
    
    def success_return(self, value):
        self.reset()
        self.func_return_value = value
        return self
    
    def success_continue(self):
        self.reset()
        self.loop_should_continue = True
        return self
    
    def success_break(self):
        self.reset()
        self.loop_should_break = True
        return self

    def failure(self, error):
        self.reset()
        self.error = error
        return self
    
    def should_return(self):
        return (
            self.error or
            self.func_return_value or
            self.loop_should_continue or
            self.loop_should_break
        )

# ---------- VALUES ------------
class Value:
    def __init__(self):
        self.set_pos()
        self.set_context()
    
    def set_pos(self, start_pos=None, end_pos=None):
        self.start_pos = start_pos
        self.end_pos = end_pos
        return self

    def set_context(self, context=None):
        self.context = context
        return self

    def add(self, other):
        return None, self.illegal_operation(other)
    
    def subtract(self, other):
        return None, self.illegal_operation(other)
    
    def multiply(self, other):
        return None, self.illegal_operation(other)
    
    def divide(self, other):
        return None, self.illegal_operation(other)

    def power(self, other):
        return None, self.illegal_operation(other)
    
    def get_comparison_eq(self, other):
        return None, self.illegal_operation(other)
    
    def get_comparison_ne(self, other):
        return None, self.illegal_operation(other)
    
    def get_comparison_lt(self, other):
        return None, self.illegal_operation(other)
    
    def get_comparison_gt(self, other):
        return None, self.illegal_operation(other)
    
    def get_comparison_lte(self, other):
        return None, self.illegal_operation(other)
    
    def get_comparison_gte(self, other):
        return None, self.illegal_operation(other)
    
    def and_(self, other):
        return None, self.illegal_operation(other)
    
    def or_(self, other):
        return None, self.illegal_operation(other)
    
    def not_(self, other):
        return None, self.illegal_operation(other)
    
    def execute(self, other):
        return None, self.illegal_operation(other)
    
    def copy(self):
        raise Exception("No copy method defined")
    
    def is_true(self):
        return False
    
    def illegal_operation(self, other=None):
        if not other: other = self
        return RTError(self.start_pos, other.end_pos, "Illegal operation", self.context)

class String(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value
    
    def add(self, other):
        if isinstance(other, String):
            return String(self.value + other.value).set_context(self.context), None
        else:
            None, Value.illegal_operation(self, other)
    
    def multiply(self, other):
        if isinstance(other, Number):
            return String(self.value * other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)
    
    def get_comparison_eq(self, other):
        if isinstance(other, String):
            return Boolean(self.value == other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_ne(self, other):
        if isinstance(other, String):
            return Boolean(self.value != other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def is_true(self):
        return len(self.value) > 0
    
    def copy(self):
        copy = String(self.value)
        copy.set_pos(self.start_pos, self.end_pos)
        copy.set_context(self.context)
        return copy
    
    def __str__(self):
        return self.value
    
    def __repr__(self):
        return f'"{self.value}"'

class Number(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value
    
    def add(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self.start_pos, other.pos_end)
    
    def subtract(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self.start_pos, other.pos_end)

    def multiply(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self.start_pos, other.pos_end)
    
    def divide(self, other):
        if isinstance(other, Number):
            if other.value == 0:
                return None, RTError(other.start_pos, other.end_pos, "Division By Zero", self.context)
            return Number(self.value / other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self.start_pos, other.pos_end)
    
    def power(self, other):
        if isinstance(other, Number):
            return Number(self.value ** other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self.start_pos, other.pos_end)

    def get_comparison_eq(self, other):
        if isinstance(other, Number):
            return Boolean(self.value == other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self.start_pos, other.pos_end)

    def get_comparison_ne(self, other):
        if isinstance(other, Number):
            return Boolean(self.value != other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self.start_pos, other.pos_end)

    def get_comparison_lt(self, other):
        if isinstance(other, Number):
            return Boolean(self.value < other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self.start_pos, other.pos_end)

    def get_comparison_gt(self, other):
        if isinstance(other, Number):
            return Boolean(self.value > other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self.start_pos, other.pos_end)

    def get_comparison_lte(self, other):
        if isinstance(other, Number):
            return Boolean(self.value <= other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self.start_pos, other.pos_end)

    def get_comparison_gte(self, other):
        if isinstance(other, Number):
            return Boolean(self.value >= other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self.start_pos, other.pos_end)

    def and_(self, other):
        if isinstance(other, Number):
            return Boolean(bool(self.value) and bool(other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self.start_pos, other.pos_end)

    def or_(self, other):
        if isinstance(other, Number):
            return Boolean(bool(self.value) or bool(other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self.start_pos, other.pos_end)


    def not_(self):
        return Number(1 if self.value == 0 else 0).set_context(self.context), None
    

    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.start_pos, self.end_pos)
        copy.set_context(self.context)
        return copy
    
    def is_true(self):
        return self.value != 0

    def __repr__(self):
        return str(self.value)

class Boolean(Value):
    def __init__(self, value):
        super().__init__()
        self.value = bool(value)

    def add(self, other):
        return None, self.illegal_operation(other)

    def subtract(self, other):
        return None, self.illegal_operation(other)

    def multiply(self, other):
        return None, self.illegal_operation(other)

    def divide(self, other):
        return None, self.illegal_operation(other)

    def power(self, other):
        return None, self.illegal_operation(other)

    def and_(self, other):
        if isinstance(other, Boolean):
            return Boolean(self.value and other.value).set_context(self.context), None
        else:
            return None, self.illegal_operation(other)

    def or_(self, other):
        if isinstance(other, Boolean):
            return Boolean(self.value or other.value).set_context(self.context), None
        else:
            return None, self.illegal_operation(other)

    def not_(self):
        return Boolean(not self.value).set_context(self.context), None

    def get_comparison_eq(self, other):
        if isinstance(other, Boolean):
            return Boolean(self.value == other.value).set_context(self.context), None
        else:
            return None, self.illegal_operation(other)

    def get_comparison_ne(self, other):
        if isinstance(other, Boolean):
            return Boolean(self.value != other.value).set_context(self.context), None
        else:
            return None, self.illegal_operation(other)

    def copy(self):
        copy = Boolean(self.value)
        copy.set_pos(self.start_pos, self.end_pos)
        copy.set_context(self.context)
        return copy

    def is_true(self):
        return self.value

    def __repr__(self):
        return "true" if self.value else "false"

Number.null = String("null")
Boolean.true = Boolean(True)
Boolean.false = Boolean(False)

class List(Value):
    def __init__(self, elements):
        super().__init__()
        self.elements = elements

    def add(self, other):
        new_list = self.copy()
        new_list.elements.append(other)
        return new_list, None
    
    def subtract(self, other):
        if isinstance(other, Number):
            new_list = self.copy()
            try:
                new_list.elements.pop(other.value)
                return new_list, None
            except:
                return RTError(other.start_pos, other.end_pos, "Element could not be removed: Index is out of bounds", self.context)
        else:
            return None, Value.illegal_operation(self, other)

    def multiply(self, other):
        if isinstance(other, List):
            new_list = self.copy()
            new_list.elements.extend(other.elements)
            return new_list, None
        else:
            return None, Value.illegal_operation(self, other)

    def divide(self, other):
        if isinstance(other, Number):
            try:
                return self.elements[other.value], None
            except IndexError:
                return None, RTError(other.start_pos, other.end_pos,  "Element could not be retrieved: Index is out of bounds",  self.context)
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_eq(self, other):
        if isinstance(other, List):
            if len(self.elements) != len(other.elements):
                return Boolean(False).set_context(self.context), None

            for i in range(len(self.elements)):
                self_elem = self.elements[i]
                other_elem = other.elements[i]

                if isinstance(self_elem, Value) and isinstance(other_elem, Value):
                    eq_result, error = self_elem.get_comparison_eq(other_elem)
                    if error or not eq_result.is_true():
                        return Boolean(False).set_context(self.context), None
                elif self_elem != other_elem:
                    return Boolean(False).set_context(self.context), None

            return Boolean(True).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def get_comparison_ne(self, other):
        if isinstance(other, List):
            eq_result, error = self.get_comparison_eq(other)
            if error:
                return None, error
            return Boolean(not eq_result.is_true()).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)

    def copy(self):
        copy = List(self.elements)
        copy.set_pos(self.start_pos, self.end_pos)
        copy.set_context(self.context)
        return copy
    
    def __str__(self):
        return f"[{', '.join([str(x) for x in self.elements])}]"
    
    def __repr__(self):
        return ", ".join([str(x) for x in self.elements])

class BaseFunction(Value):
    def __init__(self, name):
        super().__init__()
        self.name = name or "<anoymous>"
    
    def generate_context(self):
        new_context = Context(self.name, self.context, self.start_pos)
        new_context.symbol_table = SymbolTable(new_context.parent.symbol_table)
        return new_context
    
    def check_args(self, arg_names, args):
        res = RTResult()

        num_args_passed = len(args)
        num_args_expected = len(arg_names)

        if num_args_passed != num_args_expected:
            return res.failure(RTError(self.start_pos, self.end_pos, f"'{self.name}' expected {num_args_expected} arguments but got {num_args_passed}", self.context))
        
        return res.success(None)

    def populate_args(self, arg_names, args, exec_ctx):
        for i in range(len(args)):
            arg_name = arg_names[i]
            arg_value = args[i]
            arg_value.set_context(exec_ctx)
            exec_ctx.symbol_table.set(arg_name, arg_value)
        return RTResult().success(None)

    def check_and_populate_args(self, arg_names, args, exec_ctx):
        res = RTResult()

        res.register(self.check_args(arg_names, args))
        if res.should_return(): return res
        self.populate_args(arg_names, args, exec_ctx)

        return res.success(None)

class Function(BaseFunction):
    def __init__(self, name, body_node, arg_names, should_auto_return):
        super().__init__(name)
        self.body_node = body_node
        self.arg_names = arg_names
        self.should_auto_return = should_auto_return
    
    def execute(self, args):
        res = RTResult()
        interpreter = Interpreter()
        exec_ctx = self.generate_context()

        res.register(self.check_and_populate_args(self.arg_names, args, exec_ctx))
        if res.should_return(): return res

        value = res.register(interpreter.visit(self.body_node, exec_ctx))
        if res.should_return() and res.func_return_value == None: return res

        ret_value = (value if self.should_auto_return else None) or res.func_return_value or Number.null
        return res.success(ret_value)

    def copy(self):
        copy = Function(self.name, self.body_node, self.arg_names, self.should_auto_return)
        copy.set_context(self.context)
        copy.set_pos(self.start_pos, self.end_pos)
        return copy

    def __repr__(self):
        return f"<function {self.name}>"

class BuiltInFunction(BaseFunction):
    def __init__(self, name):
        super().__init__(name)
    
    def execute(self, args):
        res = RTResult()
        exec_ctx = self.generate_context()

        method_name = f"exec_{self.name}"
        method = getattr(self, method_name, self.no_visit_method)

        res.register(self.check_and_populate_args(method.arg_names, args, exec_ctx))
        if res.should_return(): return res

        return_value = res.register(method(exec_ctx))
        if res.should_return(): return res

        return res.success(return_value)

    def no_visit_method(self):
        raise Exception(f"No execute_{self.name} method defined")
    
    def copy(self):
        copy = BuiltInFunction(self.name)
        copy.set_context(self.context)
        copy.set_pos(self.start_pos, self.end_pos)
        return copy
    
    def __repr__(self):
        return f"<built-in function {self.name}>"

    def exec_print(self, exec_ctx):
        print(str(exec_ctx.symbol_table.get("value")))
        return RTResult().success(Number.null)
    exec_print.arg_names = ["value"]

    def exec_print_ret(self, exec_ctx):
        return RTResult().success(String(str(exec_ctx.symbol_table.get("value"))))
    exec_print_ret.arg_names = ["value"]

    def exec_input(self, exec_ctx):
        text = input()
        return RTResult().success(String(text))
    exec_input.arg_names = []

    def exec_input_int(self, exec_ctx):
        while True: 
            text = input()
            try:
                number = int(text)
                break
            except ValueError:
                print(f"'{text}' must be an integer")
        return RTResult().success(Number(number))
    exec_input_int.arg_names = []

    def exec_clear(self, exec_ctx):
        os.system("cls" if os.name == "nt" else "clear")
        return RTResult().success(Number.null)
    exec_clear.arg_names = []

    def exec_is_number(self, exec_ctx):
        is_number = isinstance(exec_ctx.symbol_table.get("value"), Number)
        return RTResult().success(Boolean.true if is_number else Boolean.false)
    exec_is_number.arg_names = ["value"]

    def exec_is_string(self, exec_ctx):
        is_number = isinstance(exec_ctx.symbol_table.get("value"), String)
        return RTResult().success(Boolean.true if is_number else Boolean.false)
    exec_is_string.arg_names = ["value"]

    def exec_is_list(self, exec_ctx):
        is_number = isinstance(exec_ctx.symbol_table.get("value"), List)
        return RTResult().success(Boolean.true if is_number else Boolean.false)
    exec_is_list.arg_names = ["value"]

    def exec_is_function(self, exec_ctx):
        is_number = isinstance(exec_ctx.symbol_table.get("value"), BaseFunction)
        return RTResult().success(Boolean.true if is_number else Boolean.false)
    exec_is_function.arg_names = ["value"]

    def exec_append(self, exec_ctx):
        list_ = exec_ctx.symbol_table.get("list")
        value = exec_ctx.symbol_table.get("value")

        if not isinstance(list_, List):
            return RTResult().failure(RTError(self.start_pos, self.end_pos, "First argument must be list", exec_ctx))
        
        list_.elements.append(value)

        return RTResult().success(Number.null)
    exec_append.arg_names = ["list", "value"]

    def exec_pop(self, exec_ctx):
        list_ = exec_ctx.symbol_table.get("list")
        index = exec_ctx.symbol_table.get("index")

        if not isinstance(list_, List):
            return RTResult().failure(RTError(self.start_pos, self.end_pos, "First argument must be list", exec_ctx))

        if not isinstance(index, Number):
            return RTResult().failure(RTError(self.start_pos, self.end_pos, "Second argument must be number", exec_ctx))
        
        try:
            element = list_.elements.pop(index.value)
        except:
            return RTResult().failure(RTError(self.start_pos, self.end_pos, "Element could not be removed: Index is out of bounds", exec_ctx))
        
        return RTResult().success(element)
    exec_pop.arg_names = ["list", "index"]

    def exec_extend(self, exec_ctx):
        listA = exec_ctx.symbol_table.get("listA")
        listB = exec_ctx.symbol_table.get("listB")

        if not isinstance(listA, List):
            return RTResult().failure(RTError(self.start_pos, self.end_pos, "First argument must be list", exec_ctx))

        if not isinstance(listB, List):
            return RTResult().failure(RTError(self.start_pos, self.end_pos, "Second argument must be list", exec_ctx))

        listA.elements.extend(listB.elements)
        
        return RTResult().success(Number.null)
    exec_extend.arg_names = ["listA", "listB"]

    def exec_len(self, exec_ctx):
        value = exec_ctx.symbol_table.get("value")

        if not isinstance(value, (List, String)):
            return RTResult().failure(RTError(
                self.start_pos, self.end_pos,
                "Argument must be list or string",
                exec_ctx
            ))

        if isinstance(value, List):
            return RTResult().success(Number(len(value.elements)))
        if isinstance(value, String):
            return RTResult().success(Number(len(value.value)))
    exec_len.arg_names = ["value"]

    def exec_join(self, exec_ctx):
        elements = exec_ctx.symbol_table.get("elements")
        separator = exec_ctx.symbol_table.get("separator")
        
        if not isinstance(elements, List):
            return RTResult().failure(RTError(
                self.start_pos, self.end_pos,
                "First argument must be list",
                exec_ctx
            ))
        
        if not isinstance(separator, String):
            return RTResult().failure(RTError(
                self.start_pos, self.end_pos,
                "Second argument must be string",
                exec_ctx
            ))

        str_elements = [str(element) for element in elements.elements]
        return RTResult().success(String(separator.value.join(str_elements)))
    exec_join.arg_names = ["elements", "separator"]

    def exec_map(self, exec_ctx):
        elements = exec_ctx.symbol_table.get("elements")
        func = exec_ctx.symbol_table.get("func")

        if not isinstance(elements, List):
            return RTResult().failure(RTError(
                self.start_pos, self.end_pos,
                "First argument must be list",
                exec_ctx
            ))
        
        if not isinstance(func, Function):
            return RTResult().failure(RTError(
                self.start_pos, self.end_pos,
                "Second argument must be function",
                exec_ctx
            ))

        res = RTResult()
        mapped_elements = []

        for element in elements.elements:
            func_exec_ctx = func.generate_context()
            func_exec_ctx.symbol_table.set(func.arg_names[0], element)
            result = res.register(func.execute([element]))
            if res.should_return(): return res
            mapped_elements.append(result)

        return RTResult().success(List(mapped_elements))
    exec_map.arg_names = ["elements", "func"]

    def exec_run(self, exec_ctx):
        fn = exec_ctx.symbol_table.get("filename")

        if not isinstance(fn, String):
            return RTResult().failure(RTError(
                self.start_pos, self.end_pos,
                "Argument must be string"
            ))
        
        fn = fn.value

        try:
            with open(fn, "r") as f:
                script = f.read()
        except Exception as e:
            return RTResult().failure(RTError(
                self.start_pos, self.end_pos,
                f"Failed to load script '{fn}'\n{str(e)}"
            ))
        
        _, error = run(fn, script)

        if error:
            return RTResult().failure(RTError(
                self.start_pos, self.end_pos,
                f"Failed to finish executing script '{fn}'\n{error.as_string()}",
                exec_ctx
            ))
        
        return RTResult().success(Number.null)
    exec_run.arg_names = ["filename"]

BuiltInFunction.print       = BuiltInFunction("print")
BuiltInFunction.print_ret   = BuiltInFunction("print_ret")
BuiltInFunction.input       = BuiltInFunction("input")
BuiltInFunction.input_int   = BuiltInFunction("input_int")
BuiltInFunction.clear       = BuiltInFunction("clear")
BuiltInFunction.is_number   = BuiltInFunction("is_number")
BuiltInFunction.is_string   = BuiltInFunction("is_string")
BuiltInFunction.is_list     = BuiltInFunction("is_list")
BuiltInFunction.is_function = BuiltInFunction("is_function")
BuiltInFunction.append      = BuiltInFunction("append")
BuiltInFunction.pop         = BuiltInFunction("pop")
BuiltInFunction.extend      = BuiltInFunction("extend")
BuiltInFunction.len         = BuiltInFunction("len")
BuiltInFunction.join        = BuiltInFunction("join")
BuiltInFunction.map         = BuiltInFunction("map")
BuiltInFunction.run         = BuiltInFunction("run")

# ---------- CONTEXT -----------
class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table= None

# ------- SYMBOL TABLE ---------
class SymbolTable:
    def __init__(self, parent=None):
        self.symbols = {}
        self.parent = parent

    def get(self, name):
        value = self.symbols.get(name, None)
        if value == None and self.parent:
            return self.parent.get(name)
        return value
    
    def set(self, name, value):
        self.symbols[name] = value
    
    def remove(self, name):
        del self.symbols[name]

# -------- INTERPRETER ---------
class Interpreter:
    def visit(self, node, context):
        method_name = f"visit_{type(node).__name__}"
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)
    
    def no_visit_method(self, node):
        raise Exception(f"No visit_{type(node).__name__} method defined")
    
    def visit_NumberNode(self, node, context):
        return RTResult().success(Number(node.token.value).set_context(context).set_pos(node.start_pos, node.end_pos))
    
    def visit_StringNode(self, node, context):
        return RTResult().success(String(node.token.value).set_context(context).set_pos(node.start_pos, node.end_pos))

    def visit_ListNode(self, node, context):
        res = RTResult()
        elements = []

        for element_node in node.element_nodes:
            elements.append(res.register(self.visit(element_node, context)))
            if res.should_return(): return res

        return res.success(List(elements).set_context(context).set_pos(node.start_pos, node.end_pos))

    def visit_BinaryOperatorNode(self, node, context):
        res = RTResult()
        left = res.register(self.visit(node.left_node, context))
        if res.should_return(): return res
        right = res.register(self.visit(node.right_node, context))
        if res.should_return(): return res

        if node.operator_token.type == TT_PLUS:
            result, error = left.add(right)
        elif node.operator_token.type == TT_MINUS:
            result, error = left.subtract(right)
        elif node.operator_token.type == TT_MUL:
            result, error = left.multiply(right)
        elif node.operator_token.type == TT_DIV:
            result, error = left.divide(right)
        elif node.operator_token.type == TT_POW:
            result, error = left.power(right)
        elif node.operator_token.type == TT_EE:
            result, error = left.get_comparison_eq(right)
        elif node.operator_token.type == TT_NE:
            result, error = left.get_comparison_ne(right)
        elif node.operator_token.type == TT_LT:
            result, error = left.get_comparison_lt(right)
        elif node.operator_token.type == TT_GT:
            result, error = left.get_comparison_gt(right)
        elif node.operator_token.type == TT_LTE:
            result, error = left.get_comparison_lte(right)
        elif node.operator_token.type == TT_GTE:
            result, error = left.get_comparison_gte(right)
        elif node.operator_token.matches(TT_KEYWORD, "AND"):
            result, error = left.and_(right)
        elif node.operator_token.matches(TT_KEYWORD, "OR"):
            result, error = left.or_(right)

        return res.failure(error) if error else res.success(result.set_pos(node.start_pos, node.end_pos))
    
    def visit_UnaryOperatorNode(self, node, context):
        result = RTResult()
        number = result.register(self.visit(node.node, context))
        if result.error: return result

        error = None

        if node.operator_token.type == TT_MINUS:
            number, error = number.multiply(Number(-1))
        elif node.operator_token.matches(TT_KEYWORD, "NOT"):
            number, error = number.not_()
        
        return result.failure(error) if error else result.success(number.set_pos(node.start_pos, node.end_pos))

    def visit_VarAccessNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value
        value = context.symbol_table.get(var_name)

        if value is None:
            return res.failure(RTError(node.start_pos, node.end_pos, f"'{var_name}' is not defined", context))
        
        value = value.copy().set_pos(node.start_pos, node.end_pos).set_context(context)
        return res.success(value)
    
    def visit_VarAssignNode(self, node, context):
        res = RTResult()
        var_name = node.var_name_tok.value

        if var_name in {"TRUE", "FALSE", "NULL", "PI"}:
            return res.failure(RTError(node.start_pos, node.end_pos, f"Cannot reassign global constant '{var_name}'", context))

        value = res.register(self.visit(node.value_node, context))
        if res.should_return(): return res

        context.symbol_table.set(var_name, value)
        return res.success(value)
    
    def visit_IfNode(self, node, context):
        res = RTResult()

        for condition, expr, should_return_null in node.cases:
            condition_value = res.register(self.visit(condition, context))
            if res.should_return(): return res

            if condition_value.is_true():
                expr_value = res.register(self.visit(expr, context))
                if res.should_return(): return res
                return res.success(Number.null if should_return_null else expr_value)
            
        if node.else_case:
            expr, should_return_null = node.else_case
            else_value = res.register(self.visit(expr, context))
            if res.should_return(): return res
            return res.success(Number.null if should_return_null else else_value)
        
        return res.success(Number.null)
    
    def visit_ForNode(self, node, context):
        res = RTResult()
        elements = []

        start_value = res.register(self.visit(node.start_value_node, context))
        if res.should_return(): return res

        end_value = res.register(self.visit(node.end_value_node, context))
        if res.should_return(): return res

        if node.step_value_node:
            step_value = res.register(self.visit(node.step_value_node, context))
            if res.should_return(): return res
        else:
            step_value = Number(1)
        
        i = start_value.value

        if step_value.value >= 0:
            condition = lambda: i < end_value.value
        else:
            condition = lambda: i > end_value.value
        
        while condition():
            context.symbol_table.set(node.var_name_tok.value, Number(i))
            i += step_value.value

            value = res.register(self.visit(node.body_node, context))
            if res.should_return() and res.loop_should_continue == False and res.loop_should_break == False: return res

            if res.loop_should_continue:
                continue

            if res.loop_should_break:
                break

            elements.append(value)

        return res.success(
            Number.null if node.should_return_null else
            List(elements).set_context(context).set_pos(node.start_pos, node.end_pos)
        )

    def visit_WhileNode(self, node, context):
        res = RTResult()
        elements = []

        while True:
            condition = res.register(self.visit(node.condition_node, context))
            if res.should_return(): return res

            if not condition.is_true():
                break

            value = res.register(self.visit(node.body_node, context))
            if res.should_return() and res.loop_should_continue == False and res.loop_should_break == False: return res

            if res.loop_should_continue:
                continue

            if res.loop_should_break:
                break

            elements.append(value)

        return res.success(
            Number.null if node.should_return_null else
            List(elements).set_context(context).set_pos(node.start_pos, node.end_pos)
        )
    
    def visit_FuncDefNode(self, node, context):
        res = RTResult()

        func_name = node.var_name_tok.value if node.var_name_tok else None
        body_node = node.body_node
        arg_names = [arg_name.value for arg_name in node.arg_name_toks]
        func_value = Function(func_name, body_node, arg_names, node.should_auto_return).set_context(context).set_pos(node.start_pos, node.end_pos)

        if func_name: 
            context.symbol_table.set(func_name, func_value)
        
        return res.success(func_value)
    
    def visit_FuncCallNode(self, node, context):
        res = RTResult()
        args = []

        value_to_call = res.register(self.visit(node.node_to_call, context))
        if res.should_return(): return res
        value_to_call = value_to_call.copy().set_pos(node.start_pos, node.end_pos)

        for arg_node in node.arg_nodes:
            args.append(res.register(self.visit(arg_node, context)))
            if res.should_return(): return res

        return_value = res.register(value_to_call.execute(args))
        if res.should_return(): return res
        return_value = return_value.copy().set_pos(node.start_pos, node.end_pos).set_context(context)
        return res.success(return_value)

    def visit_ReturnNode(self, node, context):
        res = RTResult()

        if node.node_to_return:
            value = res.register(self.visit(node.node_to_return, context))
            if res.should_return(): return res
        else:
            value = Number.null
        
        return res.success_return(value)
    
    def visit_ContinueNode(self, node, context):
        return RTResult().success_continue()
    
    def visit_BreakNode(self, node, context):
        return RTResult().success_break()

# ------------ RUN -------------
global_symbol_table = SymbolTable()
global_symbol_table.set("NULL", Number.null)
global_symbol_table.set("TRUE", Boolean.true)
global_symbol_table.set("FALSE", Boolean.false)
global_symbol_table.set("PI", Number(3.14159265358979323))
global_symbol_table.set("PRINT", BuiltInFunction.print)
global_symbol_table.set("PRINT_RET", BuiltInFunction.print_ret)
global_symbol_table.set("INPUT", BuiltInFunction.input)
global_symbol_table.set("INPUT_INT", BuiltInFunction.input_int)
global_symbol_table.set("CLEAR", BuiltInFunction.clear)
global_symbol_table.set("CLS", BuiltInFunction.clear)
global_symbol_table.set("IS_NUM", BuiltInFunction.is_number)
global_symbol_table.set("IS_STR", BuiltInFunction.is_string)
global_symbol_table.set("IS_LIST", BuiltInFunction.is_list)
global_symbol_table.set("IS_FUNC", BuiltInFunction.is_function)
global_symbol_table.set("APPEND", BuiltInFunction.append)
global_symbol_table.set("POP", BuiltInFunction.pop)
global_symbol_table.set("EXTEND", BuiltInFunction.extend)
global_symbol_table.set("LEN", BuiltInFunction.len)
global_symbol_table.set("JOIN", BuiltInFunction.join)
global_symbol_table.set("MAP", BuiltInFunction.map)
global_symbol_table.set("RUN", BuiltInFunction.run)

def run(file_name, text):
    # generate tokens
    lexer = Lexer(file_name, text)
    tokens, error = lexer.create_tokens()
    if error: return None, error

    # generate AST
    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error

    # run program
    interpreter = Interpreter()
    context = Context("<program>")
    context.symbol_table = global_symbol_table
    result = interpreter.visit(ast.node, context)

    result.value = repr(result.value)
    if "null" in result.value:
        result.value = result.value.replace("null", "")
        result.value = result.value.replace(",", "")
        result.value = result.value.strip()

    return result.value, result.error

if __name__ == "__main__":
    if len(sys.argv) > 3:
        print("Error: Only one argument is allowed.")
        sys.exit(1)
    elif len(sys.argv) == 2:
        try:
            open(sys.argv[1])
        except FileNotFoundError:
            print(f"Error: File '{sys.argv[1]}' not found")
            sys.exit(1)
        
        run(sys.argv[1], f'RUN("{sys.argv[1]}")')