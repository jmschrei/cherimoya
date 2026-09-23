# cherimoya_cli install-skill command
# Author: Jacob Schreiber <jmschreiber91@gmail.com>


def run(args):
	import os
	import shutil

	source = os.path.join(os.path.dirname(os.path.dirname(
		os.path.abspath(__file__))), "skills", "cherimoya")

	if not os.path.isdir(source):
		raise FileNotFoundError(
			"Bundled skill not found at {}; the package may be installed "
			"without its data files.".format(source))

	if args.directory is not None:
		skills_dir = os.path.expanduser(args.directory)
	else:
		skills_dir = os.path.expanduser(os.path.join("~", ".claude", "skills"))

	os.makedirs(skills_dir, exist_ok=True)
	dest = os.path.join(skills_dir, "cherimoya")

	if os.path.lexists(dest):
		if not args.force:
			raise FileExistsError(
				"A skill already exists at {}. Re-run with --force to "
				"overwrite it.".format(dest))

		if os.path.islink(dest) or os.path.isfile(dest):
			os.remove(dest)
		else:
			shutil.rmtree(dest)

	if args.symlink:
		os.symlink(source, dest)
		print("Symlinked Cherimoya skill:\n  {} -> {}".format(dest, source))
	else:
		shutil.copytree(source, dest,
			ignore=shutil.ignore_patterns(".ipynb_checkpoints"))
		print("Installed Cherimoya skill to:\n  {}".format(dest))

	print("Restart Claude Code (or reload skills) to pick it up.")
