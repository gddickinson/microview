#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 09:31:36 2024

@author: george
"""

import os
import ast

def find_imports(directory):
    imports = set()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                with open(os.path.join(root, file), 'r') as f:
                    try:
                        tree = ast.parse(f.read())
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for n in node.names:
                                    imports.add(n.name)
                            elif isinstance(node, ast.ImportFrom):
                                imports.add(node.module)
                    except:
                        print(f"Could not parse {file}")
    return imports

def list_py_files(directory):
    py_files = set()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                py_files.add(os.path.splitext(file)[0])
    return py_files

directory = '/Users/george/Desktop/biologicalSimulator/microview'
imports = find_imports(directory)
py_files = list_py_files(directory)

potentially_unused = py_files - imports - {'microview', 'global_vars', 'menu_manager', 'window_manager', 'image_processor', 'plugin_base'}

print("Potentially unused files:")
for file in potentially_unused:
    print(file)
