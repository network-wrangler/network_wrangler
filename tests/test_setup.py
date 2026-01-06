"""Test that the package can be setup and imported."""

import subprocess

from network_wrangler import WranglerLogger


def test_setup(request):
    """Create virtual environment and test that network wrangler can be installed and imported."""
    WranglerLogger.info(f"--Starting: {request.node.name}")
    WranglerLogger.debug("Creating virtual environment...")
    venv_name = "wranglertest"
    subprocess.run(["python3", "-m", "venv", f"{venv_name}"], check=True)
    WranglerLogger.debug("Created virtual environment.\nInstalling Wrangler...")
    install_process = subprocess.run([f"{venv_name}/bin/pip", "install", "-e", "."], check=True)
    WranglerLogger.debug(f"Installed Wrangler.\n{install_process.stdout}")
    pip_list_process = subprocess.run(
        [f"{venv_name}/bin/pip", "list"], capture_output=True, text=True, check=False
    )
    WranglerLogger.debug(f"Venv contents:\n{pip_list_process.stdout}")
    WranglerLogger.debug("Testing import...")

    # Capture output and error messages
    import_process = subprocess.run(
        [f"{venv_name}/bin/python", "-c", "import network_wrangler"],
        capture_output=True,
        text=True,
        check=False,
    )

    if import_process.returncode != 0:
        WranglerLogger.error(f"Import failed with error:\n{import_process.stderr}")
        raise subprocess.CalledProcessError(
            import_process.returncode,
            import_process.args,
            output=import_process.stdout,
            stderr=import_process.stderr,
        )

    WranglerLogger.debug("Import successful.")


def test_setup_testingenv(request):
    """Create virtual environment and test that network wrangler can be installed and imported."""
    WranglerLogger.info(f"--Starting: {request.node.name}")
    venv_name = "wranglertest"
    WranglerLogger.debug("Creating virtual environment for testing...")
    subprocess.run(["python3", "-m", "venv", f"{venv_name}"], check=True)
    WranglerLogger.debug("Created virtual environment.\nInstalling Wrangler...")
    install_process = subprocess.run(
        [f"{venv_name}/bin/pip", "install", "-e", ".[tests]"], check=True
    )
    WranglerLogger.debug(f"Installed Wrangler.\n{install_process.stdout}")
    pip_list_process = subprocess.run(
        [f"{venv_name}/bin/pip", "list"], capture_output=True, text=True, check=False
    )
    WranglerLogger.debug(f"Venv contents:\n{pip_list_process.stdout}")
    WranglerLogger.debug("Testing import...")

    # Capture output and error messages
    import_process = subprocess.run(
        [f"{venv_name}/bin/python", "-c", "import network_wrangler"],
        capture_output=True,
        text=True,
        check=False,
    )

    if import_process.returncode != 0:
        WranglerLogger.error(f"Import failed with error:\n{import_process.stderr}")
        raise subprocess.CalledProcessError(
            import_process.returncode,
            import_process.args,
            output=import_process.stdout,
            stderr=import_process.stderr,
        )

    WranglerLogger.debug("Import successful.")
