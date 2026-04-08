class VariableManagement:
    def __init__(self, parent):
        self.parent = parent
        self.shared_variables = {}

    def push_variables(self, variables):
        self.shared_variables.update(variables)
        if self.parent.console_operations and self.parent.console_operations.console:
            self.parent.console_operations.console.push_variables(variables)
        self.parent.logger.info(f"Updated shared variables: {', '.join(variables.keys())}")

    def get_variable(self, name):
        return self.shared_variables.get(name)
