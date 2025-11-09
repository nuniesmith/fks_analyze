import logging
import re
import datetime
import typer
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
import ast
import json
import subprocess
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#!/usr/bin/env python3
"""
FKS Repository Analysis Tool
Generates comprehensive reports about repository structure, files, and metrics.
"""

import os
import sys
from pathlib import Path


def run_analysis(root_dir: str, output_dir: str = "reports") -> None:
    """
    Main analysis function.
    
    Args:
        root_dir: Root directory to analyze
        output_dir: Directory to save reports
    """
    print(f"Starting analysis of {root_dir}...")
    print(f"Reports will be saved to {output_dir}/")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    def is_excluded(path: Path, exclude_patterns: List[str]) -> bool:
        return any(pat in str(path) for pat in exclude_patterns)
    
    def collect_filtered_files(target_dir: Path, exclude_patterns: List[str]) -> List[Path]:
        filter_ext = r'\.(py|rs|md|txt|sh|bash|yml|yaml|toml|json|ini|cfg|env)$'
        additional_patterns = r'(Dockerfile|docker-compose.*|Cargo.toml|Cargo.lock|pyproject.toml|requirements.*\.txt|k8s.*\.ya?ml|\.gitignore|\.github.*)'
        filtered_files = []
        for file_path in tqdm(list(target_dir.rglob('*')), desc="Collecting files"):
            if file_path.is_file() and not is_excluded(file_path.relative_to(target_dir), exclude_patterns):
                if re.search(filter_ext, file_path.name) or re.search(additional_patterns, file_path.name):
                    filtered_files.append(file_path.relative_to(target_dir))
        return sorted(filtered_files)
    
    def generate_tree(target_dir: Path, exclude_patterns: List[str]) -> str:
        def recursive_tree(current_dir: Path, prefix: str = '') -> List[str]:
            lines = []
            contents = sorted([p for p in current_dir.iterdir() if not is_excluded(p.relative_to(target_dir), exclude_patterns)])
            for i, path in enumerate(contents):
                is_last = i == len(contents) - 1
                connector = '└── ' if is_last else '├── '
                lines.append(prefix + connector + path.name)
                if path.is_dir():
                    extension = '   ' if is_last else '│   '
                    lines.extend(recursive_tree(path, prefix + extension))
            return lines
        return '\n'.join([target_dir.name] + recursive_tree(target_dir))
    
    def extract_imports_classes(file_str: str, pattern: str) -> List[str]:
        # Placeholder: Implement as needed, e.g., regex for imports/classes
        return []
    
    def extract_from_files(files: List[Path], pattern: str) -> List[str]:
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(extract_imports_classes, str(file), pattern) for file in files]
            for future in as_completed(futures):
                results.extend(future.result())
        return results
    
    def generate_mermaid_workflow(category: str, files: List[Path]) -> str:
        if category == "github":
            mermaid = "graph TD\n"
            for file in files:
                try:
                    data = yaml.safe_load(file.read_text())
                    jobs = data.get('jobs', {})
                    for job_name, job in jobs.items():
                        needs = job.get('needs', [])
                        if isinstance(needs, str):
                            needs = [needs]
                        for need in needs:
                            mermaid += f" {need} --> {job_name}\n"
                except Exception as e:
                    logger.warning(f"Failed to parse {file}: {e}")
            return mermaid or "graph TD\n A[No workflows found]"
        elif category == "docker":
            mermaid = "graph TD\n"
            for file in files:
                if "docker-compose" in file.name:
                    try:
                        data = yaml.safe_load(file.read_text())
                        services = data.get('services', {})
                        for svc_name, svc in services.items():
                            depends = svc.get('depends_on', [])
                            if isinstance(depends, dict):
                                depends = [d for d in depends if depends[d].get('condition') != 'service_healthy']
                            for dep in depends:
                                mermaid += f" {dep} --> {svc_name}\n"
                    except Exception as e:
                        logger.warning(f"Failed to parse {file}: {e}")
            return mermaid or "graph TD\n A[No compose files found]"
        return ""
    
    def generate_mermaid_project_overview(target_dir: Path) -> str:
        mermaid = "graph TD\n"
        microservices = [d.name for d in target_dir.iterdir() if d.is_dir() and d.name.startswith('fks_')]
        for ms in microservices:
            mermaid += f" {ms}[{ms}]\n"
        # Infer dependencies (expand as needed)
        mermaid += " fks_ai --> fks_api\n fks_api --> fks_data\n fks_data --> fks_training\n"
        return mermaid
    
    def generate_mermaid_logic_flow(file: Path) -> str:
        if file.suffix != '.py':
            return ""
        try:
            tree = ast.parse(file.read_text())
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            mermaid = "graph TD\n"
            for i in range(len(functions) - 1):
                mermaid += f" {functions[i]} --> {functions[i+1]}\n"  # Simple; use pyan for better call graphs
            return mermaid
        except:
            return ""
    
    def render_mermaid_to_image(mmd_path: Path):
        try:
            subprocess.run(['mmdc', '-i', str(mmd_path), '-o', str(mmd_path.with_suffix('.png'))], check=True)
        except Exception as e:
            logger.error(f"Failed to render {mmd_path}: {e}. Install mermaid-cli?")
    
    def find_empty_files_and_dirs(target_dir: Path, exclude_patterns: List[str]) -> Dict[str, List[Path]]:
        empty_files = []
        empty_dirs = []
        for path in target_dir.rglob('*'):
            if is_excluded(path.relative_to(target_dir), exclude_patterns):
                continue
            if path.is_file() and path.stat().st_size == 0:
                empty_files.append(path.relative_to(target_dir))
            elif path.is_dir() and not any(path.iterdir()):
                empty_dirs.append(path.relative_to(target_dir))
        return {'files': empty_files, 'dirs': empty_dirs}
    
    def check_broken_files(files: List[Path]) -> List[Dict[str, str]]:
        broken = []
        for file in tqdm(files, desc="Checking broken files"):
            abs_file = file.resolve()
            try:
                if abs_file.suffix == '.py':
                    ast.parse(abs_file.read_text())
                elif abs_file.suffix in ('.yml', '.yaml'):
                    yaml.safe_load(abs_file.read_text())
                elif abs_file.suffix == '.json':
                    json.loads(abs_file.read_text())
                elif abs_file.suffix in ('.sh', '.bash'):
                    subprocess.check_output(['bash', '-n', str(abs_file)])
            except Exception as e:
                broken.append({'file': str(file.relative_to(target_dir)), 'error': str(e)})
        return broken
    
    def generate_tasks(issues: Dict[str, List]) -> str:
        tasks = "# AI Agent Tasks\n\nUse these to assign to agents for fixes.\n"
        for category, items in issues.items():
            tasks += f"## {category}\n"
            for item in items:
                priority = 'High' if 'syntax' in item['error'].lower() else 'Medium'
                tasks += f"- [ ] Fix {category.lower()}: {item['file']} (Priority: {priority})\n Details: {item['error']}\n Suggested Action: Review and populate/fix code.\n"
        return tasks
    
    target_dir = target_dir.resolve()
    if not target_dir.exists():
        logger.error("Target directory does not exist")
        raise typer.Exit(code=1)
    
    filtered_files = collect_filtered_files(target_dir, exclude_patterns)
    # File structure
    tree_text = generate_tree(target_dir, exclude_patterns)
    (output / 'file_structure.txt').write_text(tree_text)
    # Empty/Broken
    empties = find_empty_files_and_dirs(target_dir, exclude_patterns)
    broken = check_broken_files([target_dir / f for f in filtered_files])
    issues = {'Empty Files': [{'file': str(f), 'error': 'File is empty'} for f in empties['files']],
              'Empty Dirs': [{'file': str(d), 'error': 'Directory is empty'} for d in empties['dirs']],
              'Broken Files': [{'file': b['file'], 'error': b['error']} for b in broken]}
    # Summary (expand with counts, etc.)
    summary = f"FKS CODE ANALYSIS SUMMARY\n=================\nDirectory: {target_dir}\nGenerated on: {datetime.datetime.now()}\n"
    summary += f"Empty Files: {len(empties['files'])}\nEmpty Dirs: {len(empties['dirs'])}\nBroken Files: {len(broken)}\n"
    (output / 'summary.txt').write_text(summary)
    # Tasks
    tasks_text = generate_tasks(issues)
    (output / 'tasks.md').write_text(tasks_text)
    # Mermaid
    github_files = [f for f in target_dir.rglob('.github/workflows/*.y*ml') if not is_excluded(f.relative_to(target_dir), exclude_patterns)]
    docker_files = [f for f in target_dir.rglob('docker-compose*.y*ml') if not is_excluded(f.relative_to(target_dir), exclude_patterns)]
    github_mmd = output / 'github_workflows.mmd'
    docker_mmd = output / 'docker_workflows.mmd'
    project_mmd = output / 'project_overview.mmd'
    github_mmd.write_text(generate_mermaid_workflow("github", github_files))
    docker_mmd.write_text(generate_mermaid_workflow("docker", docker_files))
    project_mmd.write_text(generate_mermaid_project_overview(target_dir))
    fks_apps = [d for d in target_dir.iterdir() if d.name.startswith('fks_') and d.is_dir() and not is_excluded(d.relative_to(target_dir), exclude_patterns)]
    for app in fks_apps:
        py_files = [f for f in app.rglob('*.py') if not is_excluded(f.relative_to(target_dir), exclude_patterns)]
        for py in py_files:
            logic_mmd = generate_mermaid_logic_flow(py)
            if logic_mmd:
                logic_path = output / f"{py.stem}_logic.mmd"
                logic_path.write_text(logic_mmd)
                if render:
                    render_mermaid_to_image(logic_path)
    if render:
        render_mermaid_to_image(github_mmd)
        render_mermaid_to_image(docker_mmd)
        render_mermaid_to_image(project_mmd)
    # Linting (example; expand)
    if lint:
        issues['Lint Errors'] = []
        for file in [target_dir / f for f in filtered_files if f.suffix == '.py']:
            try:
                subprocess.check_output(['flake8', str(file)])
            except Exception as e:
                issues['Lint Errors'].append({'file': str(file.relative_to(target_dir)), 'error': str(e)})
        # Update tasks if lint errors found
        tasks_text = generate_tasks(issues)
        (output / 'tasks.md').write_text(tasks_text)
        summary += f"Lint Errors: {len(issues['Lint Errors'])}\n"
        (output / 'summary.txt').write_text(summary)
    # Daily Diff (assume previous output is in a fixed dir or last timestamp)
    # Add logic here...
    logger.info("Analysis complete! Reports in %s", output)

if __name__ == '__main__':
    typer.run(main)