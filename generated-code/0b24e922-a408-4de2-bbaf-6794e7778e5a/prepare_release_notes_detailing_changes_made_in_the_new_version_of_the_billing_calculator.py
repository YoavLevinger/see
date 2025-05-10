```python
class ReleaseNotes:
    def __init__(self, version, changes):
        self.version = version
        self.changes = changes

    def to_markdown(self):
        notes = "\n# Release Notes v{}".format(self.version)
        notes += "\n\n## Changes\n"
        for change in self.changes:
            notes += "- {}".format(change)
        return notes

def update_release_notes(old_version, new_version, changes):
    if old_version and new_version:
        release_notes = ReleaseNotes(new_version, changes)
        previous_release_notes = ReleaseNotes(old_version, ['Previous release notes can be found here.'])
        return [previous_release_notes.to_markdown(), release_notes.to_markdown()]
    elif old_version:
        return [ReleaseNotes(old_version, ['Previous release notes can be found here.']).to_markdown()]
    else:
        return [ReleaseNotes(new_version, changes).to_markdown()]
```