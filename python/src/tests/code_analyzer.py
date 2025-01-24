import ast
import sys
from typing import Dict, List, Set, Tuple
from pathlib import Path
import networkx as nx
from dataclasses import dataclass

@dataclass
class ModuleInfo:
    imports: Set[str]
    imports_from: Dict[str, Set[str]]
    definitions: Set[str]
    type_hints: Set[str]

class CodebaseAnalyzer:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.modules: Dict[str, ModuleInfo] = {}
        self.import_graph = nx.DiGraph()
        
    def analyze_file(self, file_path: Path) -> ModuleInfo:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
            
        imports = set()
        imports_from = {}
        definitions = set()
        type_hints = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.add(name.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module not in imports_from:
                    imports_from[node.module] = set()
                for name in node.names:
                    imports_from[node.module].add(name.name)
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                definitions.add(node.name)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.annotation, ast.Name):
                type_hints.add(node.annotation.id)
                
        return ModuleInfo(imports, imports_from, definitions, type_hints)

    def build_import_graph(self):
        python_files = list(self.root_dir.rglob("*.py"))
        
        for file_path in python_files:
            module_name = str(file_path.relative_to(self.root_dir)).replace("/", ".").replace(".py", "")
            self.modules[module_name] = self.analyze_file(file_path)
            
        for module_name, info in self.modules.items():
            self.import_graph.add_node(module_name)
            
            # Add edges for direct imports
            for imp in info.imports:
                if imp in self.modules:
                    self.import_graph.add_edge(module_name, imp)
                    
            # Add edges for from imports
            for from_module, names in info.imports_from.items():
                if from_module in self.modules:
                    self.import_graph.add_edge(module_name, from_module)

    def find_cycles(self) -> List[List[str]]:
        try:
            return list(nx.simple_cycles(self.import_graph))
        except nx.NetworkXNoCycle:
            return []

    def find_missing_imports(self) -> Dict[str, Set[str]]:
        missing = {}
        for module_name, info in self.modules.items():
            missing_imports = set()
            
            # Check direct imports
            for imp in info.imports:
                if imp not in self.modules and not self._is_external_package(imp):
                    missing_imports.add(imp)
                    
            # Check from imports
            for from_module in info.imports_from:
                if from_module not in self.modules and not self._is_external_package(from_module):
                    missing_imports.add(from_module)
                    
            if missing_imports:
                missing[module_name] = missing_imports
                
        return missing

    def find_undefined_types(self) -> Dict[str, Set[str]]:
        undefined = {}
        for module_name, info in self.modules.items():
            undefined_types = set()
            
            for type_hint in info.type_hints:
                if not self._is_type_defined(type_hint, info):
                    undefined_types.add(type_hint)
                    
            if undefined_types:
                undefined[module_name] = undefined_types
                
        return undefined

    def _is_external_package(self, package_name: str) -> bool:
        common_externals = {
            'typing', 'dataclasses', 'abc', 'enum', 'datetime', 'pathlib',
            'uuid', 'json', 'asyncio', 'logging', 'os', 'sys',
            'anthropic', 'motor', 'pydantic', 'chromadb', 'transformers',
            'torch', 'numpy', 'spacy', 'networkx'
        }
        return package_name.split('.')[0] in common_externals

    def _is_type_defined(self, type_name: str, module_info: ModuleInfo) -> bool:
        # Check if type is defined in the module
        if type_name in module_info.definitions:
            return True
            
        # Check if type is imported
        for imports in module_info.imports_from.values():
            if type_name in imports:
                return True
                
        # Check if type is a builtin
        builtin_types = {'str', 'int', 'float', 'bool', 'list', 'dict', 'set', 'tuple', 'Any', 'Optional'}
        return type_name in builtin_types

    def analyze(self) -> Tuple[List[List[str]], Dict[str, Set[str]], Dict[str, Set[str]]]:
        """
        Perform full analysis of the codebase.
        Returns:
            - List of import cycles
            - Dict of missing imports by module
            - Dict of undefined types by module
        """
        self.build_import_graph()
        return (
            self.find_cycles(),
            self.find_missing_imports(),
            self.find_undefined_types()
        )

    def print_analysis(self):
        cycles, missing_imports, undefined_types = self.analyze()
        
        print("\n=== MAX+ Codebase Analysis ===\n")
        
        # Print import cycles
        print("Import Cycles:")
        if cycles:
            for cycle in cycles:
                print(f"  - {' -> '.join(cycle + [cycle[0]])}")
        else:
            print("  No import cycles found.")
            
        print("\nMissing Imports:")
        if missing_imports:
            for module, missing in missing_imports.items():
                print(f"  {module}:")
                for imp in missing:
                    print(f"    - {imp}")
        else:
            print("  No missing imports found.")
            
        print("\nUndefined Types:")
        if undefined_types:
            for module, types in undefined_types.items():
                print(f"  {module}:")
                for type_name in types:
                    print(f"    - {type_name}")
        else:
            print("  No undefined types found.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python code_analyzer.py <path_to_max_plus_root>")
        sys.exit(1)
        
    analyzer = CodebaseAnalyzer(sys.argv[1])
    analyzer.print_analysis()