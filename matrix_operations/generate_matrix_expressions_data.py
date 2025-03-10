import re
import math
import operator
import time
import json
import os
import numpy as np
import scipy.linalg
import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
class ExpressionNode:
    def __init__(self, operator_name, left_child_input_shape, right_child_input_shape, output_shape, output_run_time, output_total_run_time, left_child, right_child, left_child_name=None, right_child_name=None):
        self.operator_name = operator_name
        self.left_child_input_shape = left_child_input_shape
        self.right_child_input_shape = right_child_input_shape
        self.output_shape = output_shape
        self.output_run_time = output_run_time
        self.output_total_run_time = output_total_run_time
        self.left_child = left_child
        self.right_child = right_child
        self.left_child_name = left_child_name
        self.right_child_name = right_child_name

    def to_dict(self):
        return {
            'operator_name': self.operator_name,
            'left_child_input_shape': self.left_child_input_shape,
            'right_child_input_shape': self.right_child_input_shape,
            'output_shape': self.output_shape,
            'output_run_time': self.output_run_time,
            'output_total_run_time': self.output_total_run_time,
            'left_child': self.left_child.to_dict() if self.left_child else None,
            'right_child': self.right_child.to_dict() if self.right_child else None,
            'left_child_name': self.left_child_name,
            'right_child_name': self.right_child_name
        }

    def print_tree(self, level=0):
        # print like a tree with left child name, right child name, operator name
        if self.left_child:
            self.left_child.print_tree(level + 1)
        print(' ' * 4 * level + f'{self.left_child_name} {self.right_child_name} {self.operator_name}')
        if self.right_child:
            self.right_child.print_tree(level + 1)

    def display_tree(self, level=0):
        if self is None:
            return

        print('->' * level + f'{self.left_child_name} {self.right_child_name} {self.operator_name}')

        self.left_child.display_tree(level + 1) if self.left_child else None
        self.right_child.display_tree(level + 1) if self.right_child else None

    def return_outputs_as_list_postfix(self, outcomes_parallel=False):
        outputs = []
        
        if self.left_child:
            outputs.extend(self.left_child.return_outputs_as_list_postfix(outcomes_parallel=outcomes_parallel))
        
        if self.right_child:
            outputs.extend(self.right_child.return_outputs_as_list_postfix(outcomes_parallel=outcomes_parallel))
        
        if outcomes_parallel:
            if self.output_run_time is not None:
                outputs.append(self.output_run_time)
        else:
            if self.output_total_run_time is not None:
                outputs.append(self.output_total_run_time)
        
        return outputs

    def return_outputs_as_list_preorder(self, outcomes_parallel=False):
        outputs = []
        
        if outcomes_parallel:
            if self.output_run_time is not None:
                outputs.append(self.output_run_time)
        else:
            if self.output_total_run_time is not None:
                outputs.append(self.output_total_run_time)
        
        if self.left_child:
            outputs.extend(self.left_child.return_outputs_as_list_preorder(outcomes_parallel=outcomes_parallel))
        
        if self.right_child:
            outputs.extend(self.right_child.return_outputs_as_list_preorder(outcomes_parallel=outcomes_parallel))
        
        return outputs

    def get_depth(self):
        if self.left_child is None and self.right_child is None:
            return 1
        left_depth = self.left_child.get_depth() if self.left_child else 0
        right_depth = self.right_child.get_depth() if self.right_child else 0
        return max(left_depth, right_depth) + 1

    
    @classmethod
    def from_dict(cls, data):
        operator_name = data['operator_name']
        left_child_input_shape = data['left_child_input_shape']
        right_child_input_shape = data['right_child_input_shape']
        output_shape = data['output_shape']
        output_run_time = data['output_run_time']
        output_total_run_time = data['output_total_run_time']
        left_child_name = data['left_child_name']
        right_child_name = data['right_child_name']
        
        left_child = cls.from_dict(data['left_child']) if data['left_child'] else None
        right_child = cls.from_dict(data['right_child']) if data['right_child'] else None
        
        return cls(
            operator_name,
            left_child_input_shape,
            right_child_input_shape,
            output_shape,
            output_run_time,
            output_total_run_time,
            left_child,
            right_child,
            left_child_name,
            right_child_name
        )

    
        


def construct_expression_tree_from_dict(node_dict):
    if node_dict:
        left_child = construct_expression_tree_from_dict(node_dict['left_child'])
        right_child = construct_expression_tree_from_dict(node_dict['right_child'])
        return ExpressionNode(node_dict['operator_name'], node_dict['left_child_input_shape'], node_dict['right_child_input_shape'], node_dict['output_shape'], node_dict['output_run_time'], node_dict['output_total_run_time'], left_child, right_child, node_dict['left_child_name'], node_dict['right_child_name'])
    else:
        return None



class ExpressionEvaluator:
    def __init__(self):

        self.operators = {
                '+': np.add,         # Matrix addition
                '-': np.subtract,    # Matrix subtraction
                'dot': np.multiply,  # Element-wise multiplication
                '*': np.matmul,      # Matrix multiplication
                'TR': np.transpose,  # Matrix transpose
                'inv': np.linalg.inv,  # Matrix inverse
                'det': np.linalg.det,  # Matrix determinant
                'trace': np.trace,   # Matrix trace
                'norm': np.linalg.norm,  # Matrix norm
                'SVD': np.linalg.svd,  # Singular Value Decomposition
                'LU': scipy.linalg.lu,  # LU factorization
                'QR': np.linalg.qr    # QR factorization
                
                
        }
        

    ary_operators = {
        '+': 2,
        '-': 2,
        '*': 2,
        'dot': 2,
        'inv': 1,
        'det': 1,
        'trace': 1,
        'norm': 2,
        'TR': 1,
        'SVD': 2,
        'LU': 2,
        'QR': 2
        }
    def replace_variables(self, expression, v, w):
        # don't replace, max or clamp

        replaced_expr = expression.replace('v', str(v)).replace('w', str(w))

        # "m{a}x" back to "max"
        # replaced_expr = re.sub(r'm\{(\w+)\}x', r'max', replaced_expr)
        return replaced_expr

    def parse(self, expression, N, variables_dict):
        # variables = {'v': v, 'w': w}
        # replace x and y in the expression with their values except for the operators max, min, clamp
        # expression = self.replace_variables(expression, v, w)
        # print(expression)
        


        # for var, value in variables.items():
        #     expression = expression.replace(var, str(value))
        
        # Tokenize the expression using regex
        tokens = re.findall(r'(\b\w*[\.]?\w+\b|[\+\-\*\/\#\(\)])', expression)
        # print(tokens)

        
        
        # Convert infix notation to postfix notation using the Shunting Yard algorithm
        output_queue = self.shunting_yard(tokens)
        
        return output_queue

    def evaluate(self, expression, N, variables_dict):
        output_queue = self.parse(expression, N, variables_dict)
        try:
            result, runtimes, operation_features, result_json, result_tree = self.evaluate_postfix(output_queue, N, variables_dict)
        except ValueError as e:
            raise e
            return
        
        return result, runtimes, operation_features, result_json, result_tree

    def shunting_yard(self, tokens):
        # include all the operators and their precedence
        precedence = {
        '+': 2,
        '-': 2,
        '*': 3,
        'dot': 5,
        'inv': 5,
        'det': 5,
        'trace': 5,
        'norm': 5,
        'TR': 5,
        'SVD': 5,
        'LU': 5,
        'QR': 5
        }
        # precedence = {'+': 1, '-': 1, '*': 2, '/': 2, '^': 3}

        output_queue = []
        operator_stack = []

        for token in tokens:
            if token in precedence:
                while (operator_stack and operator_stack[-1] in precedence and
                       precedence[token] <= precedence[operator_stack[-1]]):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(token)
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack[-1] != '(':
                    output_queue.append(operator_stack.pop())
                operator_stack.pop()  # Discard the '('
            else:
                output_queue.append(token)

        while operator_stack:
            output_queue.append(operator_stack.pop())

        return output_queue

    def parse_expression(self, expr):
        tokens = re.findall(r'(\b\w*[\.]?\w+\b|[\+\-\*\/\(\)\[\]])', expr)
        stack = []
        for token in tokens:
            if token == '(':
                stack.append(token)
            elif token == ')':
                subexpr = []
                while stack[-1] != '(':
                    subexpr.append(stack.pop())
                stack.pop()  # Remove '('
                subexpr.reverse()
                stack.append(subexpr)
            # elif token == '[':
            #     stack.append(token)
            # elif token == ']':
            #     index = stack.pop()
            #     stack[-1].append(index)
            else:
                stack.append(token)
        return stack[0]

    def evaluate_postfix(self, postfix_tokens, N, variables_dict):
        result_json = {"result": []}
        stack = []
        runtimes = []
        operation_features = []
        expr_tree_stack = []

        total_time = 0
        # print(postfix_tokens)
        loop_start_time = time.perf_counter_ns()
        for token in postfix_tokens:
            if token in self.operators:
                num_args = self.ary_operators[token]
                args = [stack.pop() for _ in range(num_args)]
                args = args[::-1]
                left_child = None
                right_child = None
                left_child_name = None
                right_child_name = None

                
                # if isinstance(args[0], str) and args[0] in variables_dict:
                #     left_child = None
                # else:
                #     left_child = expr_tree_stack.pop()
                # if len(args) > 1:
                #     if args[1] in variables_dict:
                #         right_child = None
                #     else:
                #         right_child = expr_tree_stack.pop()

                # expr_tree_args = [expr_tree_stack.pop() for _ in range(num_args)]
                # expr_tree_args = expr_tree_args[::-1]

                for i, arg in enumerate(args):
                    if isinstance(arg, str) and arg not in ['fro', '0', '1', '2', 'inf']:
                        if arg in variables_dict:
                            args[i] = variables_dict[arg][:N, :N]
                            if i == 0:
                                left_child = None
                                left_child_name = arg
                            else:
                                right_child = None
                                right_child_name = arg
                    elif isinstance(arg, str) and arg in ['fro', '0', '1', '2', 'inf']:
                        if i == 0:
                            left_child = None
                            left_child_name = arg
                        else:
                            right_child = None
                            right_child_name = arg
                    else:
                        if i == 0:
                            left_child = expr_tree_stack.pop()
                            left_child_name = left_child.operator_name
                        else:
                            i == 1
                            right_child = expr_tree_stack.pop()
                            right_child_name = right_child.operator_name
                       

                start_time = time.perf_counter_ns()
                if token in ["SVD", "LU", "QR"]:
                    result = self.operators[token](args[0])[int(args[1])]
                    args = [args[0]]
                else:
                    if token == "*":
                        if isinstance(args[0], np.ndarray) and isinstance(args[1], np.ndarray):
                            result = self.operators[token](args[0], args[1])
                            matmul_type = 'both_arrays'
                        elif isinstance(args[0], np.ndarray):
                            result = self.operators["dot"](args[0], args[1])
                            matmul_type = 'first_array'
                        elif isinstance(args[1], np.ndarray):
                            result = self.operators["dot"](args[0], args[1])
                            matmul_type = 'second_array'
                        else:
                            result = np.float64(args[0]) * np.float64(args[1])
                            matmul_type = 'no_array'
                    else:
                        if token == "norm":
                            result = self.operators[token](args[0], ord=args[1])
                            args = [args[0]]
                        else:
                            result = self.operators[token](*args)
                end_time = time.perf_counter_ns()

                op_runtime = end_time - start_time
                total_time += op_runtime
                runtimes.append((token, op_runtime))
                stack.append(result)

                input_features = {}
                for i, arg in enumerate(args):
                    if isinstance(arg, np.ndarray):
                        input_features[f'arg_shape_{i}'] = arg.size
                    else:
                        input_features[f'arg_shape_{i}'] = 1
                for i, arg in enumerate(args):
                    if isinstance(arg, np.ndarray):
                        input_features[f'arg_norm_{i}'] = np.linalg.norm(arg)
                    else:
                        input_features[f'arg_norm_{i}'] = arg
                    if token == "*":
                        input_features['matmul_type'] = matmul_type
                if isinstance(result, np.ndarray):
                    input_features['result_shape'] = result.size
                    input_features['result_norm'] = np.linalg.norm(result)
                else:
                    input_features['result_shape'] = 1
                    input_features['result_norm'] = result

                operation_features.append({'input_features': input_features})
                result_json["result"].append({'operation': token, 'input_features': input_features, 'runtime': op_runtime})

                left_child_input_shape = input_features['arg_shape_0']
                right_child_input_shape = input_features['arg_shape_1'] if "arg_shape_1" in input_features else None
                output_shape = input_features['result_shape']
                # add left child if it exists
                # if argument is a variable, left child is None
                output_total_run_time = op_runtime + (left_child.output_total_run_time if left_child else 0) + (right_child.output_total_run_time if right_child else 0)
                expr_tree_node = ExpressionNode(token, left_child_input_shape, right_child_input_shape, output_shape, op_runtime, output_total_run_time, left_child, right_child, left_child_name, right_child_name)
                expr_tree_stack.append(expr_tree_node)
            else:
                stack.append(token)
                # print(token)
                # expr_tree_stack.append(ExpressionNode(token, None, None, variables_dict[token].shape, 0, None, None))

        loop_end_time = time.perf_counter_ns()
        loop_runtime = loop_end_time - loop_start_time
        runtimes.append(('loop_time', loop_runtime))
        runtimes.append(('total_time', total_time))
        # expr_tree_stack[0].display_tree()
        return stack[0], runtimes, operation_features, result_json, expr_tree_stack[0]

   
def main():
    # set seed
    np.random.seed(0)
    load_matrices = False

    evaluator = ExpressionEvaluator()
    # create results for different expressions
    results = {}
    # Here are the expressions with spaces removed between the @ operator:
    # expressions = ["SVD(R)[1] * LU(S)[0] * (T * QR(U)[0]) * (V + W) * inv(X) * dot(Y, Z) * trace(A * B) * det(C) * norm(D, 'fro')",]
    expressions = [
            "(A + B) * SVD(C)[0] * LU(D)[1] * QR(E)[1] * (F - G) * inv(H) * dot(I, J) * trace(K * L) * det(M) * norm(N, 'fro')",
            "SVD(O)[2] * (P * LU(Q)[0]) * QR(R)[0] * (S + T) * inv(U) * dot(V, W) * trace(X * Y) * det(Z) * norm(A, 'fro')",
            "(B + C) * (D * SVD(E)[1]) * LU(F)[1] * (G * QR(H)[1]) * inv(I) * dot(J, K) * trace(L * M) * det(N) * norm(O, 'fro')",
            "QR(P)[0] * SVD(Q)[0] * (R * LU(S)[0]) * (T + U) * inv(V) * dot(W, X) * trace(Y * Z) * det(A) * norm(B, 'fro')",
            "(C + D) * (E * SVD(F)[2]) * LU(G)[1] * QR(H)[1] * (I - J) * inv(K) * dot(L, M) * trace(N * O) * det(P) * norm(Q, 'fro')",
            "SVD(R)[1] * LU(S)[0] * (T * QR(U)[0]) * (V + W) * inv(X) * dot(Y, Z) * trace(A * B) * det(C) * norm(D, 'fro')",
            "(E + F) * (G * SVD(H)[0]) * LU(I)[1] * QR(J)[1] * (K - L) * inv(M) * dot(N, O) * trace(P * Q) * det(R) * norm(S, 'fro')",
            "QR(T)[0] * SVD(U)[2] * LU(V)[0] * (W * X) * inv(Y) * dot(Z, A) * trace(B * C) * det(D) * norm(E, 'fro')",
            "(F + G) * (H * SVD(I)[1]) * LU(J)[1] * QR(K)[1] * (L - M) * inv(N) * dot(O, P) * trace(Q * R) * det(S) * norm(T, 'fro')",
            "SVD(U)[0] * LU(V)[0] * QR(W)[0] * (X + Y) * inv(Z) * dot(A, B) * trace(C * D) * det(E) * norm(F, 'fro')",
            "(A + B) * C * TR(D) * dot(E, F) * inv(G) * (H + I) * J * TR(K) * dot(L, M) * inv(N) * (O + P) * Q * TR(R) * dot(S, T) * inv(U) * (V + W)",
            "2 * trace(X * Y) * dot(Z, A) * inv(B) * (C + D) * E * TR(F) * dot(G, H) * inv(I) * (J + K) * L * TR(M) * dot(N, O) * inv(P) * (Q + R) * S",
            "det(T) * (U - V) * dot(W, X) * TR(Y) * norm(Z, 'fro') * (A + B) * C * TR(D) * dot(E, F) * inv(G) * (H + I) * J * TR(K) * dot(L, M) * inv(N) * (O + P)",
            "(Q + R) * dot(S, T) * inv(U) * (V * TR(W)) * trace(X * Y) * 3 * (Z + A) * dot(B, C) * inv(D) * (E * TR(F)) * norm(G, 'fro') * (H + I) * J * TR(K) * dot(L, M) * inv(N)",
            "det(O) * (P - Q) * dot(R, S) * TR(T) * (U + V) * dot(W, X) * inv(Y) * (Z * TR(A)) * trace(B * C) * 4 * (D + E) * dot(F, G) * inv(H) * (I * TR(J)) * norm(K, 'fro') * (L + M)",
            "(N * TR(O)) * dot(P, Q) * inv(R) * (S + T) * U * TR(V) * dot(W, X) * inv(Y) * (Z + A) * B * TR(C) * dot(D, E) * inv(F) * (G + H) * I * TR(J) * dot(K, L) * inv(M) * (N + O)",
            "5 * trace(P * Q) * dot(R, S) * inv(T) * (U + V) * W * TR(X) * dot(Y, Z) * inv(A) * (B + C) * D * TR(E) * dot(F, G) * inv(H) * (I + J) * K * TR(L) * dot(M, N) * inv(O) * (P + Q)",
            "det(R) * (S - T) * dot(U, V) * TR(W) * norm(X, 'fro') * (Y + Z) * A * TR(B) * dot(C, D) * inv(E) * (F + G) * H * TR(I) * dot(J, K) * inv(L) * (M + N) * O * TR(P) * dot(Q, R) * inv(S) * (T + U)",
            "(V + W) * dot(X, Y) * inv(Z) * (A * TR(B)) * trace(C * D) * 6 * (E + F) * dot(G, H) * inv(I) * (J * TR(K)) * norm(L, 'fro') * (M + N) * O * TR(P) * dot(Q, R) * inv(S) * (T + U) * V * TR(W) * dot(X, Y) * inv(Z)",
            "det(A) * (B - C) * dot(D, E) * TR(F) * (G + H) * dot(I, J) * inv(K) * (L * TR(M)) * trace(N * O) * 7 * (P + Q) * dot(R, S) * inv(T) * (U * TR(V)) * norm(W, 'fro') * (X + Y) * Z * TR(A) * dot(B, C) * inv(D) * (E + F)",
            "(G + H) * dot(I, J) * inv(K) * (L * TR(M)) * trace(N * O) * 8 * (P + Q) * dot(R, S) * inv(T) * (U * TR(V)) * norm(W, 'fro') * (X + Y) * Z * TR(A) * dot(B, C) * inv(D) * (E + F) * G * TR(H) * dot(I, J) * inv(K)",
            "det(L) * (M - N) * dot(O, P) * TR(Q) * (R + S) * dot(T, U) * inv(V) * (W * TR(X)) * trace(Y * Z) * 9 * (A + B) * dot(C, D) * inv(E) * (F * TR(G)) * norm(H, 'fro') * (I + J) * K * TR(L) * dot(M, N) * inv(O) * (P + Q)",
            "(R + S) * dot(T, U) * inv(V) * (W * TR(X)) * trace(Y * Z) * 10 * (A + B) * dot(C, D) * inv(E) * (F * TR(G)) * norm(H, 'fro') * (I + J) * K * TR(L) * dot(M, N) * inv(O) * (P + Q) * R * TR(S) * dot(T, U) * inv(V)",
            "det(W) * (X - Y) * dot(Z, A) * TR(B) * (C + D) * dot(E, F) * inv(G) * (H * TR(I)) * trace(J * K) * 11 * (L + M) * dot(N, O) * inv(P) * (Q * TR(R)) * norm(S, 'fro') * (T + U) * V * TR(W) * dot(X, Y) * inv(Z) * (A + B)",
            "(C + D) * dot(E, F) * inv(G) * (H * TR(I)) * trace(J * K) * 12 * (L + M) * dot(N, O) * inv(P) * (Q * TR(R)) * norm(S, 'fro') * (T + U) * V * TR(W) * dot(X, Y) * inv(Z) * (A + B) * C * TR(D) * dot(E, F) * inv(G)",
            "det(H) * (I - J) * dot(K, L) * TR(M) * (N + O) * dot(P, Q) * inv(R) * (S * TR(T)) * trace(U * V) * 13 * (W + X) * dot(Y, Z) * inv(A) * (B * TR(C)) * norm(D, 'fro') * (E + F) * G * TR(H) * dot(I, J) * inv(K) * (L + M)",
            "(N + O) * dot(P, Q) * inv(R) * (S * TR(T)) * trace(U * V) * 14 * (W + X) * dot(Y, Z) * inv(A) * (B * TR(C)) * norm(D, 'fro') * (E + F) * G * TR(H) * dot(I, J) * inv(K) * (L + M) * N * TR(O) * dot(P, Q) * inv(R)"]


    v_min_value = 2
    v_max_value = 1001
    folder = "{}/data".format(ROOT_DIR)
    if not os.path.exists(folder):
        os.makedirs(folder)
    matrix_folder = "{}/matrices".format(folder)
    if not os.path.exists(matrix_folder):
        os.makedirs(matrix_folder)
    for i, expr in tqdm.tqdm(enumerate(expressions[:1])):
        # print(i, expr)
        results = {}
        results[expr] = []
        matrix_filename = "{}/results_matrices_expr_{}.npy".format(matrix_folder, i)
        # find all the variables in the expression
        variables = re.findall(r'\b\w*[\.]?\w+\b', expr)
        # remove operators from the list of variables
        variables = [var for var in variables if var not in evaluator.operators]
        
        # remove 'fro', '0', '1', '2', 'inf' from the list of variables
        variables = [var for var in variables if var not in ['fro', '0', '1', '2', 'inf']]
        variables_matrix_dict = {}

        if load_matrices:
            # load the matrices
            variables_matrix_dict = np.load(matrix_filename, allow_pickle=True)
            variables_matrix_dict = variables_matrix_dict.item()
        else:
            variables_matrix_dict = {}
            for v in variables:
                # generate random matrices
                variables_matrix_dict[v] = np.random.rand(v_max_value, v_max_value)

        for v_log in tqdm.tqdm(range(v_min_value, v_max_value)):
            r_dict = {}
            N = v_log
            
            # parsed_expr = evaluator.parse(expr, N, variables_matrix_dict)
            parsed_expr = evaluator.parse_expression(expr)
            # print(parsed_expr)
            # root = evaluator.build_expression_tree(parsed_expr, variables_matrix_dict, N)
            result, runtimes, operation_features, result_json, root = evaluator.evaluate(expr, N, variables_matrix_dict)
            r_dict["expr"] = expr
            r_dict["matrix_size"] = N
            r_dict["total_runtime"] = runtimes[-1][1]
            # r_dict["result"] = result 
            r_dict["result_json"] = result_json
            r_dict["expression_tree"] = root.to_dict()
            results[expr].append(r_dict)
               

        filename = "{}/results_expr_{}.json".format(folder, i)
        # matrix_filename = "{}/results_matrices_expr_{}.npy".format(folder, i)
       
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)

        # save the matrices to a numpy file
        np.save(matrix_filename, variables_matrix_dict)

        


if __name__ == "__main__":
    main()
