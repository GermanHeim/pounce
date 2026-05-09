#!/usr/bin/env python3
"""Extract all roptions->Add*Option(...) and SetRegisteringCategory(...) calls
from a list of upstream Ipopt .cpp files and emit a JSON list of
{op, name, type, args, category, file, lineno} records preserving order.
"""
import re, sys, json, pathlib

# Parse an arg list at offset `i` in `text`, where text[i] == '('.
# Return (args_list, end_offset) where end_offset is index AFTER the matching ')'.
# Args are split on commas at depth 0; whitespace-trimmed; raw text preserved.
def parse_args(text, i):
    assert text[i] == '('
    depth = 0
    args = []
    cur = []
    j = i
    n = len(text)
    in_str = False
    while j < n:
        c = text[j]
        if in_str:
            cur.append(c)
            if c == '\\' and j+1 < n:
                cur.append(text[j+1]); j += 2; continue
            if c == '"':
                in_str = False
            j += 1; continue
        if c == '"':
            in_str = True; cur.append(c); j += 1; continue
        if c == '/' and j+1 < n and text[j+1] == '/':
            # line comment; skip to newline
            k = text.find('\n', j)
            if k == -1: k = n
            j = k; continue
        if c == '/' and j+1 < n and text[j+1] == '*':
            k = text.find('*/', j+2)
            if k == -1: break
            j = k + 2; continue
        if c == '(':
            depth += 1
            if depth == 1:
                j += 1
                continue
            else:
                cur.append(c); j += 1; continue
        if c == ')':
            depth -= 1
            if depth == 0:
                if cur or args:
                    args.append(''.join(cur).strip())
                return args, j + 1
            cur.append(c); j += 1; continue
        if c == ',' and depth == 1:
            args.append(''.join(cur).strip())
            cur = []
            j += 1; continue
        cur.append(c); j += 1
    return args, j

KIND_RE = re.compile(
    r'roptions\s*->\s*('
    r'AddLowerBoundedNumberOption|AddLowerBoundedIntegerOption|'
    r'AddBoundedNumberOption|AddBoundedIntegerOption|'
    r'AddStringOption[0-9]*|AddNumberOption|AddIntegerOption|AddBoolOption|'
    r'SetRegisteringCategory'
    r')\b'
)

def preprocess_ifdefs(src):
    """Resolve a small set of upstream preprocessor branches the way the
    default Ipopt build resolves them. We choose:
      * `#ifndef IPOPT_SINGLE` → keep body  (double-precision build)
      * `#ifdef IPOPT_SINGLE`  → drop body
      * `#ifdef COINHSL_HAS_METIS` → keep body
      * `#ifdef PARDISO_LIB`   → keep #else branch (default build does not set it)
    Plus expand `IPOPT_SHAREDLIBEXT` to "so".
    """
    out = src.replace('IPOPT_SHAREDLIBEXT', '"so"')

    def keep_first(m):  return m.group(1)
    def keep_second(m): return m.group(2)
    def drop(m):        return ''

    out = re.sub(r'#ifndef\s+IPOPT_SINGLE\b[^\n]*\n(.*?)\n\s*#else\b[^\n]*\n(.*?)\n\s*#endif\b[^\n]*',
                 keep_first, out, flags=re.DOTALL)
    out = re.sub(r'#ifndef\s+IPOPT_SINGLE\b[^\n]*\n(.*?)\n\s*#endif\b[^\n]*',
                 keep_first, out, flags=re.DOTALL)
    out = re.sub(r'#ifdef\s+IPOPT_SINGLE\b[^\n]*\n(.*?)\n\s*#else\b[^\n]*\n(.*?)\n\s*#endif\b[^\n]*',
                 keep_second, out, flags=re.DOTALL)
    out = re.sub(r'#ifdef\s+IPOPT_SINGLE\b[^\n]*\n(.*?)\n\s*#endif\b[^\n]*',
                 drop, out, flags=re.DOTALL)
    out = re.sub(r'#ifdef\s+COINHSL_HAS_METIS\b[^\n]*\n(.*?)\n\s*#else\b[^\n]*\n(.*?)\n\s*#endif\b[^\n]*',
                 keep_first, out, flags=re.DOTALL)
    out = re.sub(r'#ifdef\s+COINHSL_HAS_METIS\b[^\n]*\n(.*?)\n\s*#endif\b[^\n]*',
                 keep_first, out, flags=re.DOTALL)
    out = re.sub(r'#ifdef\s+PARDISO_LIB\b[^\n]*\n(.*?)\n\s*#else\b[^\n]*\n(.*?)\n\s*#endif\b[^\n]*',
                 keep_second, out, flags=re.DOTALL)
    out = re.sub(r'#ifdef\s+FUNNY_MA57_FINT\b[^\n]*\n(.*?)\n\s*#else\b[^\n]*\n(.*?)\n\s*#endif\b[^\n]*',
                 keep_second, out, flags=re.DOTALL)
    return out

def extract(path):
    src = path.read_text()
    src = preprocess_ifdefs(src)
    out = []
    cur_cat = ""
    for m in KIND_RE.finditer(src):
        kind = m.group(1)
        end = m.end()
        # find '(' (skip ws)
        k = end
        while k < len(src) and src[k] in ' \t\n':
            k += 1
        if k >= len(src) or src[k] != '(':
            continue
        args, after = parse_args(src, k)
        # compute lineno of match start
        lineno = src.count('\n', 0, m.start()) + 1
        if kind == 'SetRegisteringCategory':
            # Two-arg form is priority-setting in `RegisterAllIpoptOptions`;
            # those calls don't transition state for any nearby option
            # registrations -- skip them to keep output clean.
            if len(args) != 1:
                continue
            cat_arg = args[0]
            cur_cat = cat_arg
            out.append({'op': 'cat', 'value': cat_arg, 'file': str(path), 'line': lineno})
        else:
            out.append({'op': 'add', 'kind': kind, 'args': args, 'category': cur_cat, 'file': str(path), 'line': lineno})
    return out

if __name__ == '__main__':
    base = pathlib.Path(sys.argv[1])
    files = sys.argv[2:]
    all_records = []
    for f in files:
        p = base / f
        all_records.extend(extract(p))
    json.dump(all_records, sys.stdout, indent=2)
