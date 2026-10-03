# Coding Standards

- **PEP 8** compliance – all Python code should be formatted with `black` and linted with `flake8`. Use `--max-line-length=88` for consistency.
- **Type hints** – every public function/class should expose explicit type annotations. Use `typing.Protocol` where appropriate.
- **Docstrings** – follow the Google‑style docstring format; include a short summary, parameter descriptions, return type and examples.
- **Imports** – group imports into three sections: standard library, third‑party, local modules, separated by blank lines. Avoid `import *`.
- **Naming conventions** – use snake_case for functions/variables, PascalCase for classes, ALL_CAPS for constants.
- **Testing support** – expose any helper functions or fixtures in a dedicated `_test_support.py`. Do not import tests into the main package to avoid circular imports.
- **Cline integration** – place `# cfn` directives (if needed) at the top of files that will be processed by Cline. Use `# @cli: generate_docs` for documentation generation.

Adhering to these rules ensures maximum compatibility with automated tools and reduces merge conflicts.