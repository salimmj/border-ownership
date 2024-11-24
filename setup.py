import re
from pathlib import Path


def replace_in_file(file_path: Path, old_text: str, new_text: str) -> None:
    """Replace text in a file.

    Args:
    ----
        file_path: The path to the file to replace text in.
        old_text: The text to replace.
        new_text: The new text to replace the old text with.

    """
    with Path(file_path).open() as file:
        content = file.read()
    content = content.replace(old_text, new_text)
    with Path(file_path).open("w") as file:
        file.write(content)


def main() -> None:
    """Set up the project."""
    project_name = input("Enter your project name: ")

    default_project_name = "pyprojecttemplate"

    # Update pyproject.toml
    replace_in_file(
        "pyproject.toml",
        default_project_name,
        project_name,
    )

    # Move the src directory to the project name
    src_dir = Path(default_project_name)
    new_src_dir = Path(project_name)
    src_dir.rename(new_src_dir)

    # Update README.md
    with Path("README.md").open() as file:
        content = file.read()
    content = re.sub(r"# VSCode Dev Container: .*", f"# {project_name}", content)
    with Path("README.md").open("w") as file:
        file.write(content)

    print(f"Project {project_name} has been set up successfully!")  # noqa: T201


if __name__ == "__main__":
    main()
