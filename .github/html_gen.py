import html
import re
import shutil
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

# Result files are named <machine>_<version>.pdf, where <machine> is usually
# <cpu>_<os>. Early uploads kept the tool's default "benchmark_result" infix.
VERSION_RE = re.compile(r"\d+(?:\.\d+)*")
LEGACY_INFIX = "_benchmark_result"

# Machines that were uploaded under an older name, mapped onto their current
# name so that all results of a machine are listed together.
MACHINE_ALIASES = {
    "i7-13700K": "i7-13700K_linux",
    "rpi5": "armv8-rpi5b",
}


def parse_result_name(pdf_file):
    """Return (machine, version key) for a result PDF; version key is None if absent."""
    machine, _, version = pdf_file.stem.rpartition("_")
    if not machine or not VERSION_RE.fullmatch(version):
        return pdf_file.stem, None
    machine = machine.removesuffix(LEGACY_INFIX)
    machine = MACHINE_ALIASES.get(machine, machine)
    return machine, tuple(int(part) for part in version.split("."))


def version_order(entry):
    pdf_file, version = entry
    return version is not None, version or (), pdf_file.name


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
        h3 {
            color: #555;
            font-size: 1rem;
            margin: 1rem 0 0.25rem;
        }
        ul {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        li {
            margin: 0.25rem 0;
            padding: 0.1rem 0.5rem;
            border-left: 3px solid transparent;
        }
        li.latest {
            background-color: #e6f4ea;
            border-left-color: #2e7d32;
            font-weight: bold;
        }
        .badge {
            display: inline-block;
            margin-left: 0.5rem;
            padding: 0 0.4rem;
            border-radius: 0.25rem;
            background-color: #2e7d32;
            color: #fff;
            font-size: 0.75rem;
            vertical-align: middle;
        }
        .note {
            color: #666;
            margin: 0.25rem 0;
        }
        .note .badge {
            margin-left: 0;
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
        <p class="note">Results for the newest ssrJSON release are highlighted as <span class="badge">latest</span>.</p>
        <p class="note">Newest ssrJSON release on GitHub: <a id="ssrjson-latest" href="https://github.com/Antares0982/ssrJSON/releases/latest" target="_blank">see releases</a></p>
"""
    )

    results_dir = Path("results")
    subdirs = sorted(d for d in results_dir.iterdir() if d.is_dir())
    if not subdirs:
        f.write("<p>No benchmark results available.</p>\n")
    else:
        for subdir in subdirs:
            f.write(f"<h2>{html.escape(subdir.name)}</h2>\n")

            machines = defaultdict(list)
            for pdf_file in subdir.glob("*.pdf"):
                machine, version = parse_result_name(pdf_file)
                machines[machine].append((pdf_file, version))

            if not machines:
                f.write("<p>No PDF files available.</p>\n")
                continue

            for machine in sorted(machines, key=str.casefold):
                entries = sorted(machines[machine], key=version_order, reverse=True)

                f.write(f"<h3>{html.escape(machine.replace('_', ' '))}</h3>\n<ul>\n")
                for pdf_file, version in entries:
                    relative_path = pdf_file.relative_to(results_dir)
                    link = (
                        f"<a href='{html.escape(quote(relative_path.as_posix()))}' "
                        f"target='_blank'>{html.escape(pdf_file.name)}</a>"
                    )
                    if version is not None:
                        version_attr = html.escape(".".join(map(str, version)))
                        f.write(f"<li data-version='{version_attr}'>{link}</li>\n")
                    else:
                        f.write(f"<li>{link}</li>\n")

                    dest_path = output_dir / relative_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(pdf_file, dest_path)
                f.write("</ul>\n")

    f.write(
        """    </main>
    <script>
        const RELEASE_API = "https://api.github.com/repos/Antares0982/ssrJSON/releases/latest";
        // Cache the release in localStorage to stay clear of the GitHub API rate limit.
        const CACHE_KEY = "ssrjson-latest-release";
        const CACHE_TTL_MS = 60 * 60 * 1000;

        function normalizeVersion(tag) {
            return tag.replace(/^v/i, "").split(".").map(Number).join(".");
        }

        function applyRelease(release) {
            const link = document.getElementById("ssrjson-latest");
            link.textContent = release.tag_name;
            link.href = release.html_url;

            const version = normalizeVersion(release.tag_name);
            document.querySelectorAll("li[data-version]").forEach((item) => {
                const isLatest = item.dataset.version === version;
                const badge = item.querySelector(".badge");
                item.classList.toggle("latest", isLatest);
                if (isLatest && !badge) {
                    item.insertAdjacentHTML("beforeend", "<span class='badge'>latest</span>");
                } else if (!isLatest && badge) {
                    badge.remove();
                }
            });
        }

        let cached = null;
        try {
            cached = JSON.parse(localStorage.getItem(CACHE_KEY));
        } catch (e) {}
        if (cached && cached.release) {
            applyRelease(cached.release);
        }
        if (!cached || !(Date.now() - cached.time < CACHE_TTL_MS)) {
            fetch(RELEASE_API)
                .then((response) => (response.ok ? response.json() : Promise.reject(response.status)))
                .then((json) => {
                    const release = { tag_name: json.tag_name, html_url: json.html_url };
                    applyRelease(release);
                    try {
                        localStorage.setItem(CACHE_KEY, JSON.stringify({ time: Date.now(), release }));
                    } catch (e) {}
                })
                .catch(() => {});
        }
    </script>
</body>
</html>
"""
    )
