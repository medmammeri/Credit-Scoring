# this file tells pip and python about the package

from typing import Dict, List

import pathlib

import setuptools

# author = Mohammed Rafik Mammeri
# licence = MIT
# licence_file = LICENSE
# platforms = unix, linux, osx, cygwin, win32
# version = 0.0.1


if __name__ == "__main__":
    project_name = "Credit Scoring Model"
    project_description = "Credit Scoring Modelling project based on public data"
    package_name = "Credit_Scoring"
    requirements_path = pathlib.Path(__file__).parent / "requirements"

    install_requires = requirements_path.joinpath("prod").read_text().splitlines()
    dev_requires = requirements_path.joinpath("dev").read_text().splitlines()

    extras_requires: Dict[str, List[str]] = {"dev": dev_requires}

    extras_requires_path = requirements_path / "extras"
    if extras_requires_path.is_dir():
        all_requires = []
        for extra in extras_requires_path.iterdir():
            extras_requires[extra.name] = extra.read_text().splitlines()
            all_requires.extend(extras_requires[extra.name])

        extras_requires["all"] = all_requires
        extras_requires["dev"].extend(all_requires)

    setuptools.setup(
        name=project_name,
        description=project_description,
        python_requires=">=3.9",
        packages=setuptools.find_packages(exclude=["tests"]),
        install_requires=install_requires,
        setup_requires=["setuptools_scm"],
        extras_require=extras_requires,
        # define the app inside __main__ as the entrypoint for project
        entry_points={
            "console_scripts": (f"{project_name}={package_name}.__main__:app",),
        },
        include_package_data=True,
        zip_safe=False,
    )
