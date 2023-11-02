import re
from .skillset import SkillSet

class MiniSpecInterpreter:
    low_level_skillset: SkillSet = None
    high_level_skillset: SkillSet = None
    def __init__(self):
        self.env = {}
        if MiniSpecInterpreter.low_level_skillset is None or MiniSpecInterpreter.high_level_skillset is None:
            raise Exception('MiniSpecInterpreter: Skillset is not initialized')

    def get_env_value(self, var):
        if var not in self.env:
            raise Exception(f'Variable {var} is not defined')
        return self.env[var]

    def split_statements(self, code):
        statements = []
        stack = []
        start = 0

        for i, char in enumerate(code):
            if char == '{':
                stack.append('{')
            elif char == '}':
                stack.pop()
                if len(stack) == 0:
                    statements.append(code[start:i+1].strip())
                    start = i + 1
            elif char == ';' and len(stack) == 0:
                statements.append(code[start:i].strip())
                start = i + 1

        if start < len(code):
            statements.append(code[start:].strip())

        return [s for s in statements if s]

    def execute(self, code):
        statements = self.split_statements(code)
        for statement in statements:
            print(f'Executing statement: {statement}')
            if not statement:
                continue
            if statement.startswith('->'):
                return self.evaluate_return(statement)
            elif statement[1:].lstrip().startswith('{'):
                result = self.execute_loop(statement)
                if result is not None:
                    return result
            elif statement.startswith('?'):
                result = self.execute_conditional(statement)
                if result is not None:
                    return result
            else:
                self.execute_function_call(statement)

    def evaluate_return(self, statement):
        _, value = statement.split('->')
        return self.evaluate_value(value.strip())
    
    def execute_loop(self, statement):
        count, program = re.match(r'(\d+)\s*\{(.+)\}', statement).groups()
        for i in range(int(count)):
            print(f'Executing loop iteration {i}')
            result = self.execute(program)
            if result is not None:
                return result

    def execute_conditional(self, statement):
        condition, program = statement.split('{')
        condition = condition[1:].strip()
        program = program[:-1]
        if self.evaluate_condition(condition):
            return self.execute(program)

    def execute_function_call(self, statement):
        if '=' in statement:
            var, func = statement.split('=')
            self.env[var.strip()] = self.call_function(func)
        else:
            self.call_function(statement)

    def evaluate_condition(self, condition) -> bool:
        if '&' in condition:
            conditions = condition.split('&')
            return all(map(self.evaluate_condition, conditions))
        if '|' in condition:
            conditions = condition.split('|')
            return any(map(self.evaluate_condition, conditions))
        var, comparator, value = re.match(r'(_\d+)\s*(==|!=|<|>)\s*(.+)', condition).groups()
        var_value = self.get_env_value(var)
        if comparator == '>':
            return var_value > self.evaluate_value(value)
        elif comparator == '<':
            return var_value < self.evaluate_value(value)
        elif comparator == '==':
            return var_value == self.evaluate_value(value)
        elif comparator == '!=':
            return var_value != self.evaluate_value(value)

    def call_function(self, func):
        name, args = re.match(r'(\w+)(?:,(.+))?', func).groups()
        if args:
            args = args.split(',')
            # replace _1, _2, ... with their values
            for i in range(0, len(args)):
                if args[i].startswith('_'):
                    args[i] = self.get_env_value(args[i])
        print(f'Calling skill {name} with args {args}')

        skill_instance = MiniSpecInterpreter.low_level_skillset.get_skill_by_abbr(name)
        if skill_instance is not None:
            return skill_instance.execute(args)

        skill_instance = MiniSpecInterpreter.high_level_skillset.get_skill_by_abbr(name)
        if skill_instance is not None:
            interpreter = MiniSpecInterpreter()
            return interpreter.execute(skill_instance.execute(args))
        raise Exception(f'Skill {name} is not defined')
    

    def evaluate_value(self, value):
        if value.isdigit():
            return int(value)
        elif value.replace('.', '', 1).isdigit():
            return float(value)
        elif value == 'True':
            return True
        elif value == 'False':
            return False
        else:
            return value.strip('\'"')