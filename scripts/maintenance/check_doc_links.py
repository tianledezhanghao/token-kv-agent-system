"""Check repository-local Markdown/HTML paths and static anchors without edits."""
from __future__ import annotations
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]


def prose(text):
    lines = []
    fence = None
    for line in text.splitlines(keepends=True):
        match = re.match(r'^\s*(`{3,}|~{3,})', line)
        if match:
            marker = match.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            lines.append('\n')
        else:
            lines.append('\n' if fence else line)
    return ''.join(lines)


class HTMLLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for key in ('href', 'src'):
            if attrs.get(key):
                self.links.append((self.getpos()[0], attrs[key]))
        for key in ('id', 'name'):
            if attrs.get(key):
                self.ids.add(attrs[key])


def anchors(path):
    text = path.read_text(errors='replace')
    parser = HTMLLinks()
    parser.feed(text)
    result = parser.ids
    if path.suffix.lower() == '.md':
        counts = {}
        for heading in re.findall(r'^\s{0,3}#{1,6}\s+(.+)$', prose(text), re.M):
            heading = re.sub(r'\[([^]]+)\]\([^)]*\)', r'\1', heading)
            heading = re.sub(r'<[^>]+>', '', heading).strip().rstrip('#').strip()
            slug = re.sub(r'[^\w\- ]', '', heading.lower()).replace(' ', '-')
            number = counts.get(slug, 0)
            counts[slug] = number + 1
            result.add(slug + (f'-{number}' if number else ''))
    return result


def links(path):
    raw = path.read_text(errors='replace')
    text = re.sub(r'`[^`\n]*`', lambda m: ' ' * len(m.group()), prose(raw)) if path.suffix.lower() == '.md' else raw
    parser = HTMLLinks()
    parser.feed(text)
    found = list(parser.links)
    if path.suffix.lower() == '.md':
        for match in re.finditer(r'!?\[[^]\n]*\]\(\s*(?:<([^>]+)>|([^\s)]+))(?:\s+["\'][^\n]*["\'])?\s*\)', text):
            found.append((text.count('\n', 0, match.start()) + 1, match.group(1) or match.group(2)))
        for match in re.finditer(r'^\s*\[[^]]+\]:\s*<?([^\s>]+)>?', text, re.M):
            found.append((text.count('\n', 0, match.start()) + 1, match.group(1)))
    return found


def check(files):
    issues, checked, cache = [], 0, {}
    for path in sorted(files):
        if not path.is_file() or path.suffix.lower() not in ('.md', '.html'):
            continue
        for line, target in links(path):
            if target.startswith(('//', 'data:', 'javascript:', 'mailto:')) or '${' in target:
                continue
            url = urlsplit(target)
            if url.scheme or url.netloc:
                continue
            checked += 1
            dest = (path.parent / unquote(url.path)).resolve() if url.path else path
            if url.path.startswith('/') and path.is_relative_to(ROOT / 'frontend'):
                dest = ROOT / 'frontend' / unquote(url.path).lstrip('/')
            reason = None
            if not dest.exists():
                reason = 'missing_path'
            elif url.fragment and dest.is_file() and dest.suffix.lower() in ('.md', '.html'):
                if dest not in cache:
                    cache[dest] = anchors(dest)
                if unquote(url.fragment) not in cache[dest]:
                    reason = 'missing_static_anchor'
            if reason:
                issues.append(dict(file=str(path.relative_to(ROOT)), line=line, target=target, reason=reason))
    return dict(files=len(files), local_links=checked, issues=issues)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--files-from', type=Path, help='UTF-8 list of repository-relative document paths')
    parser.add_argument('--include-untracked', action='store_true')
    args = parser.parse_args()
    if args.files_from:
        names = args.files_from.read_text().splitlines()
    else:
        command = ['git', 'ls-files', '-z']
        if args.include_untracked:
            command += ['--cached', '--others', '--exclude-standard']
        names = subprocess.check_output(command, cwd=ROOT).decode().split('\0')
    files = {ROOT / name for name in names if name and Path(name).suffix.lower() in ('.md', '.html')}
    report = check(files)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return bool(report['issues'])


if __name__ == '__main__':
    raise SystemExit(main())
