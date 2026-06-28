#!/usr/bin/env python3
"""Simple cluster test script.

This script scans a given folder and prints the file types found
and how many files exist for each type, along with a confirmation
message including the host and process id so you can verify it ran
on the remote resource.
"""

import os
from collections import Counter

# --- Configure the folder to scan here ---
FOLDER = r"/scratch/eande106/ZihaoJohnLi/datasets/Amanda/20260226_Cry_p001_p020_Lina/raw_images"
# -----------------------------------------


def main():
	folder = FOLDER
	if not os.path.isdir(folder):
		print(f"Error: '{folder}' is not a valid directory.")
		return

	counts = Counter()
	for entry in os.scandir(folder):
		if entry.is_file():
			ext = os.path.splitext(entry.name)[1].lower()
			counts[ext if ext else "(no extension)"] += 1

	print(f"Cluster test OK: pid={os.getpid()}, folder='{folder}'")
	print(f"Found {sum(counts.values())} file(s) across {len(counts)} type(s):")
	for ext, count in sorted(counts.items()):
		print(f"  {ext}: {count}")


if __name__ == "__main__":
	main()

