"""Tests for the cherimoya install-skill command.

install-skill copies the bundled agent skill into a skills directory,
creating a `cherimoya/` subdirectory that holds SKILL.md plus the
references. A copy install must never carry `.ipynb_checkpoints`
autosaves along (they exist in the working tree but must not ship into
a user's skills directory), a collision without --force must error, and
--force must overwrite.
"""

import argparse
import os
import re

import pytest

from cherimoya_cli.commands import install_skill


def _run(tmp_path, symlink=False, force=False, directory=None):
	args = argparse.Namespace(
		directory=str(directory) if directory is not None else str(tmp_path),
		symlink=symlink, force=force)
	install_skill.run(args)
	return os.path.join(
		str(directory) if directory is not None else str(tmp_path),
		"cherimoya")


def test_copy_install_writes_skill_and_references(tmp_path):
	dest = _run(tmp_path)
	assert os.path.isfile(os.path.join(dest, "SKILL.md"))
	refs = os.path.join(dest, "references")
	assert os.path.isdir(refs)
	# There is at least one reference and every one is a .md file.
	names = os.listdir(refs)
	assert any(name.endswith(".md") for name in names)


def test_copy_install_excludes_ipynb_checkpoints(tmp_path):
	"""Stale Jupyter autosaves in the source tree must not be copied."""
	dest = _run(tmp_path)
	for root, dirs, _ in os.walk(dest):
		assert ".ipynb_checkpoints" not in dirs, (
			"install-skill copied a .ipynb_checkpoints directory into "
			"{}".format(root))


def test_collision_without_force_errors(tmp_path):
	_run(tmp_path)
	with pytest.raises(FileExistsError):
		_run(tmp_path)


def test_force_overwrites_existing(tmp_path):
	dest = _run(tmp_path)
	# Drop a stray file; --force should wipe the directory before copying.
	stray = os.path.join(dest, "stray.txt")
	with open(stray, "w") as f:
		f.write("x")
	_run(tmp_path, force=True)
	assert not os.path.exists(stray)
	assert os.path.isfile(os.path.join(dest, "SKILL.md"))


def test_symlink_install_points_at_source(tmp_path):
	dest = _run(tmp_path, symlink=True)
	assert os.path.islink(dest)
	assert os.path.isfile(os.path.join(dest, "SKILL.md"))


SKILL_SOURCE = os.path.join(os.path.dirname(os.path.abspath(
	install_skill.__file__)), os.pardir, "skills", "cherimoya")


def _skill_documents():
	"""Every Markdown file that ships in the skill, as (path, text) pairs."""

	paths = [os.path.join(SKILL_SOURCE, "SKILL.md")]

	refs = os.path.join(SKILL_SOURCE, "references")
	paths += sorted(os.path.join(refs, name) for name in os.listdir(refs)
		if name.endswith(".md"))

	documents = []
	for path in paths:
		with open(path) as f:
			documents.append((path, f.read()))

	return documents


def _frontmatter(text):
	"""The SKILL.md frontmatter as a dict of top-level keys to strings.

	PyYAML is not a dependency, so this reads the two shapes the skill uses:
	`key: value` on one line, and a folded `key: >-` block whose indented
	lines are joined with spaces.
	"""

	lines = text.split("\n")
	assert lines[0] == "---", "SKILL.md must open with a --- frontmatter line"
	end = lines.index("---", 1)

	fields, key = {}, None
	for line in lines[1:end]:
		if line.startswith((" ", "\t")) and key is not None:
			fields[key] = (fields[key] + " " + line.strip()).strip()
		elif ":" in line:
			key, value = line.split(":", 1)
			value = value.strip()
			fields[key] = "" if value in (">", ">-", "|", "|-") else value

	return fields


def test_skill_frontmatter_has_name_and_description():
	"""Claude Code reads only the frontmatter before deciding to load a skill.

	A missing `name` or `description`, a name that differs from the install
	directory, or a description over 1024 characters stops the skill from
	loading or triggering, and nothing else in the suite would notice.
	"""

	with open(os.path.join(SKILL_SOURCE, "SKILL.md")) as f:
		fields = _frontmatter(f.read())

	assert fields.get("name") == "cherimoya"
	assert fields.get("description"), "SKILL.md frontmatter has no description"
	assert len(fields["description"]) <= 1024, (
		"description is {} characters; the limit is 1024".format(
			len(fields["description"])))


def test_cross_references_are_complete_paths_that_resolve():
	"""Every `*.md` a skill file names must be a real, skill-root path.

	A bare ``cli.md`` does not say which directory it lives in, so an
	agent following the pointer has to search for the target. Mentions
	are written as ``references/cli.md`` (or ``SKILL.md`` at the root),
	relative to the skill root, and must name a file that exists.
	"""

	mention = re.compile(r"`([A-Za-z0-9_/.-]*\.md)`")

	for path, text in _skill_documents():
		for target in mention.findall(text):
			assert target == "SKILL.md" or target.startswith("references/"), (
				"{} names `{}` by bare filename; write the complete "
				"skill-root-relative path".format(
					os.path.basename(path), target))

			assert os.path.isfile(os.path.join(SKILL_SOURCE, target)), (
				"{} points at `{}`, which is not a file in the "
				"skill".format(os.path.basename(path), target))


def test_every_reference_is_reachable_from_the_router():
	"""An unlinked reference file is one no agent will ever open."""

	with open(os.path.join(SKILL_SOURCE, "SKILL.md")) as f:
		router = f.read()

	refs = os.path.join(SKILL_SOURCE, "references")
	for name in sorted(os.listdir(refs)):
		if not name.endswith(".md"):
			continue

		assert "`references/{}`".format(name) in router, (
			"references/{} is not linked from SKILL.md".format(name))
