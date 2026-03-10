#!/usr/bin/env python3
"""Assemble an experiment directory from a template.

Usage:
    python setup_task.py <template> <target_dir>
    python setup_task.py --list

Examples:
    python setup_task.py mujoco/fetchreach my_experiment
    python setup_task.py mujoco/fetchpickplace test_pickplace001
    python setup_task.py isaac/fetchreach test_isaac001
    python setup_task.py --list
"""

import argparse
import re
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
CORE_DIR = REPO_ROOT / "core"
TEMPLATES_DIR = REPO_ROOT / "templates"


def list_templates():
    """List all available templates."""
    templates = []
    for sim_dir in sorted(TEMPLATES_DIR.iterdir()):
        if not sim_dir.is_dir():
            continue
        for task_dir in sorted(sim_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            name = f"{sim_dir.name}/{task_dir.name}"
            # Check if it's a prototype (has NotImplementedError in prepare.py)
            prepare = task_dir / "prepare.py"
            is_prototype = False
            if prepare.exists():
                content = prepare.read_text()
                if "NotImplementedError" in content:
                    is_prototype = True
            label = f"  {name}"
            if is_prototype:
                label += " (prototype)"
            templates.append(label)
    return templates


def read_time_budget(prepare_path: Path) -> int:
    """Extract TIME_BUDGET value from a prepare.py file."""
    content = prepare_path.read_text()
    match = re.search(r"^TIME_BUDGET\s*=\s*(\d+)", content, re.MULTILINE)
    if not match:
        print(f"ERROR: Could not find TIME_BUDGET in {prepare_path}", file=sys.stderr)
        sys.exit(1)
    return int(match.group(1))


def format_time_human(seconds: int) -> str:
    """Convert seconds to human-readable time string."""
    if seconds < 60:
        return f"{seconds} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    else:
        hours = seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''}"


def compute_template_vars(time_budget: int) -> dict:
    """Compute all template variables from TIME_BUDGET.

    These values are deterministic: given a TIME_BUDGET, the program.md
    output is fully determined.
    """
    time_budget_human = format_time_human(time_budget)

    # Experiment timeout: training + eval/render overhead (~1-5 min depending on task)
    if time_budget < 60:
        overhead_seconds = 80  # short tasks: ~1.5 min total
    elif time_budget <= 600:
        overhead_seconds = 120  # medium tasks: ~2 min overhead
    else:
        overhead_seconds = 300  # long tasks: ~5 min overhead
    total_seconds = time_budget + overhead_seconds
    total_minutes = total_seconds / 60
    if total_minutes < 5:
        experiment_timeout_human = f"~{total_minutes:.1f} minutes"
    else:
        experiment_timeout_human = f"~{total_minutes:.0f} minutes"

    # Kill timeout: generous cap, minimum 60 min
    if time_budget <= 600:
        kill_timeout = 60
    else:
        kill_timeout = 90

    return {
        "TIME_BUDGET_HUMAN": time_budget_human,
        "EXPERIMENT_TIMEOUT_HUMAN": experiment_timeout_human,
        "KILL_TIMEOUT_MINUTES": str(kill_timeout),
    }


def render_template(template_content: str, variables: dict) -> str:
    """Replace {{PLACEHOLDER}} tokens in template content."""
    result = template_content
    for key, value in variables.items():
        result = result.replace(f"{{{{{key}}}}}", value)
    return result


def setup_task(template_name: str, target_dir: Path, program_path: Path = None):
    """Assemble an experiment directory from a template."""
    template_path = TEMPLATES_DIR / template_name
    if not template_path.is_dir():
        print(f"ERROR: Template '{template_name}' not found at {template_path}", file=sys.stderr)
        print(f"\nAvailable templates:", file=sys.stderr)
        for t in list_templates():
            print(t, file=sys.stderr)
        sys.exit(1)

    if target_dir.exists():
        print(f"ERROR: Target directory '{target_dir}' already exists", file=sys.stderr)
        sys.exit(1)

    # Create target directory
    target_dir.mkdir(parents=True)

    # Step 1: Copy all files from core/
    for item in CORE_DIR.iterdir():
        if item.is_file():
            shutil.copy2(str(item), str(target_dir / item.name))

    # Step 2: Copy all files from template (overwriting core files)
    for item in template_path.rglob("*"):
        if item.is_file():
            rel = item.relative_to(template_path)
            dest = target_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(item), str(dest))

    # Step 3: Read TIME_BUDGET from the assembled prepare.py
    prepare_path = target_dir / "prepare.py"
    time_budget = read_time_budget(prepare_path)

    # Step 4: Render program.md.template → program.md
    template_file = target_dir / "program.md.template"
    if template_file.exists():
        variables = compute_template_vars(time_budget)
        rendered = render_template(template_file.read_text(), variables)
        (target_dir / "program.md").write_text(rendered)
        template_file.unlink()  # Remove the template file from output

    # Override with custom program.md if provided
    if program_path is not None:
        if not program_path.exists():
            print(f"ERROR: Custom program.md not found: {program_path}", file=sys.stderr)
            sys.exit(1)
        shutil.copy2(str(program_path), str(target_dir / "program.md"))

    # Print summary
    print(f"Created experiment directory: {target_dir}")
    print(f"  Template: {template_name}")
    print(f"  Time budget: {format_time_human(time_budget)} ({time_budget}s)")
    print()
    print("Files:")
    for f in sorted(target_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name}")
    print()
    print("Next steps:")
    print(f"  cd {target_dir}")
    print(f"  git init && git add -A && git commit -m 'init'")
    print(f"  uv sync")
    print(f"  uv run prepare.py")


def main():
    parser = argparse.ArgumentParser(
        description="Assemble an experiment directory from a template",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("template", nargs="?", help="Template name (e.g. mujoco/fetchreach)")
    parser.add_argument("target", nargs="?", help="Target directory to create")
    parser.add_argument("--list", action="store_true", help="List available templates")
    parser.add_argument("--program", type=Path, help="Custom program.md to use instead of default")

    args = parser.parse_args()

    if args.list:
        print("Available templates:")
        for t in list_templates():
            print(t)
        return

    if not args.template or not args.target:
        parser.print_help()
        sys.exit(1)

    setup_task(args.template, Path(args.target), program_path=args.program)


if __name__ == "__main__":
    main()
