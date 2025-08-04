import os
import shutil
from pathlib import Path
from datetime import datetime

output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

index_file = output_dir / "index.html"
with index_file.open("w", encoding="utf-8") as f:
    f.write(
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Results</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f4f4f9;
            color: #333;
        }
        header {
            background: #333;
            color: #fff;
            padding: 1rem 0;
            text-align: center;
        }
        h1 {
            margin: 0;
        }
        main {
            padding: 1rem;
        }
        h2 {
            color: #444;
            border-bottom: 2px solid #ddd;
            padding-bottom: 0.5rem;
        }
        ul {
            list-style: none;
            padding: 0;
        }
        li {
            margin: 0.5rem 0;
        }
        a {
            text-decoration: none;
            color: #007BFF;
        }
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <header>
        <h1>ssrJSON Benchmark Results</h1>
    </header>
    <main>
"""
    )

    results_dir = Path("results")
    subdirs = sorted(results_dir.iterdir())
    if not subdirs:
        f.write("<p>No benchmark results available.</p>\n")
    else:
        for subdir in subdirs:
            if subdir.is_dir():
                f.write(f"<h2>{subdir.name}</h2>\n<ul>\n")

                # pdf name in BRAND-CPU_VERSION-*.pdf
                pdf_files = sorted(
                    subdir.glob("*.pdf"),
                    key=lambda x: x.stem.split("_")[-1],
                )

                if not pdf_files:
                    f.write("<li>No PDF files available.</li>\n")
                else:
                    for pdf_file in pdf_files:
                        cpu_name, version = pdf_file.stem.split("_")[:2]
                        relative_path = pdf_file.relative_to(results_dir)
                        f.write(
                            f"<li><a href='{relative_path}' target='_blank'>{pdf_file.name}</a></li>\n"
                        )

                        dest_path = output_dir / relative_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(pdf_file, dest_path)

                f.write("</ul>\n")

    f.write(
        """    </main>
</body>
</html>
"""
    )
