import os


def mkpath(path):
    """
    Checks if the given path is a file or a directory and creates it if it doesn't exist.

    :param path: str - The file or directory path to check and create.
    """
    if not os.path.exists(path):
        if "." in os.path.basename(
            path
        ):  # Assuming it's a file if there's an extension
            # Create the parent directories if they don't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Create an empty file
            with open(path, "w") as f:
                pass
        else:
            # Create the directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
        print(f"Created path: {path}")
    else:
        print(f"Path already exists: {path}")
