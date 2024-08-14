#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 22:47:23 2024

@author: george
"""

import ast
import astor
import vulture
import os

class DeadCodeCommenter(ast.NodeTransformer):
    def __init__(self, dead_code):
        self.dead_code = dead_code

    def visit_FunctionDef(self, node):
        if any(item.name == node.name for item in self.dead_code):
            return ast.Comment(f"# DEAD CODE: {ast.get_source_segment(self.source, node)}")
        return node

    def visit_ClassDef(self, node):
        if any(item.name == node.name for item in self.dead_code):
            return ast.Comment(f"# DEAD CODE: {ast.get_source_segment(self.source, node)}")
        return node

def comment_out_dead_code(file_path):
    # Run Vulture
    v = vulture.Vulture()
    v.scavenge([file_path])

    # Read the original file
    with open(file_path, 'r') as file:
        source = file.read()

    # Parse the AST
    tree = ast.parse(source)

    # Comment out dead code
    transformer = DeadCodeCommenter(v.get_unused_code())
    transformer.source = source
    new_tree = transformer.visit(tree)

    # Generate the new source code
    new_source = astor.to_source(new_tree)

    # Write the new source back to the file
    with open(file_path, 'w') as file:
        file.write(new_source)

    print(f"Processed {file_path}")

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                comment_out_dead_code(file_path)

# Usage
if __name__ == "__main__":
    process_directory("/path/to/your/project")
