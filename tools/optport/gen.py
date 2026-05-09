#!/usr/bin/env python3
"""Generate Rust registration code from upstream Ipopt RegisterOptions.

See module docstring at top of optport/extract.py for parsing details.
This script walks each per-class .cpp in the order defined by
`RegisterAllIpoptOptions` and emits one Rust function that mirrors
upstream's full set of registered options.
"""
import json, re, pathlib, sys, subprocess

ROOT = pathlib.Path(sys.argv[1])

CLASS_ORDER = [
    # RegisterOptions_Interfaces
    ("Interfaces/IpIpoptApplication.cpp", None,                       "IpoptApplication"),
    ("Common/IpRegOptions.cpp",           None,                       "RegisteredOptions"),
    ("Interfaces/IpTNLPAdapter.cpp",      None,                       "TNLPAdapter"),
    # RegisterOptions_Algorithm
    ("Algorithm/IpAdaptiveMuUpdate.cpp",            "Barrier Parameter Update", "AdaptiveMuUpdate"),
    ("Algorithm/IpDefaultIterateInitializer.cpp",   "Initialization",            "DefaultIterateInitializer"),
    ("Algorithm/IpAlgBuilder.cpp",                  "",                          "AlgorithmBuilder"),
    ("Algorithm/IpBacktrackingLineSearch.cpp",      "Line Search",               "BacktrackingLineSearch"),
    ("Algorithm/IpFilterLSAcceptor.cpp",            "Line Search",               "FilterLSAcceptor"),
    ("Algorithm/IpPenaltyLSAcceptor.cpp",           "Line Search",               "PenaltyLSAcceptor"),
    ("Algorithm/IpNLPScaling.cpp",                  "NLP Scaling",               "StandardScalingBase"),
    ("Algorithm/IpGradientScaling.cpp",             "NLP Scaling",               "GradientScaling"),
    ("Algorithm/IpEquilibrationScaling.cpp",        "NLP Scaling",               "EquilibrationScaling"),
    ("Algorithm/IpIpoptAlg.cpp",                    "",                          "IpoptAlgorithm"),
    ("Algorithm/IpIpoptData.cpp",                   "",                          "IpoptData"),
    ("Algorithm/IpIpoptCalculatedQuantities.cpp",   "",                          "IpoptCalculatedQuantities"),
    ("Algorithm/IpLimMemQuasiNewtonUpdater.cpp",    "Hessian Approximation",     "LimMemQuasiNewtonUpdater"),
    ("Algorithm/IpMonotoneMuUpdate.cpp",            "Barrier Parameter Update",  "MonotoneMuUpdate"),
    ("Algorithm/IpOptErrorConvCheck.cpp",           "Termination",               "OptimalityErrorConvergenceCheck"),
    ("Algorithm/IpOrigIpoptNLP.cpp",                "NLP",                       "OrigIpoptNLP"),
    ("Algorithm/IpOrigIterationOutput.cpp",         "Output",                    "OrigIterationOutput"),
    ("Algorithm/IpPDSearchDirCalc.cpp",             "Step Calculation",          "PDSearchDirCalculator"),
    ("Algorithm/IpPDFullSpaceSolver.cpp",           "Step Calculation",          "PDFullSpaceSolver"),
    ("Algorithm/IpPDPerturbationHandler.cpp",       "Step Calculation",          "PDPerturbationHandler"),
    ("Algorithm/IpProbingMuOracle.cpp",             "Barrier Parameter Update",  "ProbingMuOracle"),
    ("Algorithm/IpQualityFunctionMuOracle.cpp",     "Barrier Parameter Update",  "QualityFunctionMuOracle"),
    ("Algorithm/IpRestoConvCheck.cpp",              "Restoration Phase",         "RestoConvergenceCheck"),
    ("Algorithm/IpRestoFilterConvCheck.cpp",        "Restoration Phase",         "RestoFilterConvergenceCheck"),
    ("Algorithm/IpRestoIpoptNLP.cpp",               "Restoration Phase",         "RestoIpoptNLP"),
    ("Algorithm/IpRestoPenaltyConvCheck.cpp",       "Restoration Phase",         "RestoPenaltyConvergenceCheck"),
    ("Algorithm/IpRestoMinC_1Nrm.cpp",              "Restoration Phase",         "MinC_1NrmRestorationPhase"),
    ("Algorithm/IpWarmStartIterateInitializer.cpp", "Warm Start",                "WarmStartIterateInitializer"),
    # RegisterOptions_CGPenalty
    ("contrib/CGPenalty/IpCGSearchDirCalc.cpp",     "CG Penalty",                "CGSearchDirCalculator"),
    ("contrib/CGPenalty/IpCGPenaltyLSAcceptor.cpp", "CG Penalty",                "CGPenaltyLSAcceptor"),
    ("contrib/CGPenalty/IpCGPenaltyCq.cpp",         "CG Penalty",                "CGPenaltyCq"),
    # RegisterOptions_LinearSolvers
    ("Algorithm/LinearSolvers/IpTSymLinearSolver.cpp",    "Linear Solver",        "TSymLinearSolver"),
    ("Algorithm/LinearSolvers/IpMa27TSolverInterface.cpp","MA27 Linear Solver",   "Ma27TSolverInterface"),
    ("Algorithm/LinearSolvers/IpMa57TSolverInterface.cpp","MA57 Linear Solver",   "Ma57TSolverInterface"),
    ("Algorithm/LinearSolvers/IpMa77SolverInterface.cpp", "MA77 Linear Solver",   "Ma77SolverInterface"),
    ("Algorithm/LinearSolvers/IpMa86SolverInterface.cpp", "MA86 Linear Solver",   "Ma86SolverInterface"),
    ("Algorithm/LinearSolvers/IpMa97SolverInterface.cpp", "MA97 Linear Solver",   "Ma97SolverInterface"),
    ("Algorithm/LinearSolvers/IpMumpsSolverInterface.cpp","Mumps Linear Solver",  "MumpsSolverInterface"),
    ("Algorithm/LinearSolvers/IpPardisoSolverInterface.cpp",     "Pardiso (pardiso-project.org) Linear Solver", "PardisoSolverInterface"),
    ("Algorithm/LinearSolvers/IpPardisoMKLSolverInterface.cpp",  "Pardiso (MKL) Linear Solver",                  "PardisoMKLSolverInterface"),
    ("Algorithm/LinearSolvers/IpSpralSolverInterface.cpp",       "SPRAL Linear Solver",                          "SpralSolverInterface"),
    ("Algorithm/LinearSolvers/IpWsmpSolverInterface.cpp",        "WSMP Linear Solver",                           "WsmpSolverInterface"),
    ("Algorithm/LinearSolvers/IpIterativeWsmpSolverInterface.cpp","WSMP Linear Solver",                          "IterativeWsmpSolverInterface"),
    ("Algorithm/LinearSolvers/IpMa28TDependencyDetector.cpp",    "MA28 Linear Solver",                           "Ma28TDependencyDetector"),
]


# ---- C++ string handling -------------------------------------------------

# Match a single C string literal, capturing its raw body.
STRING_TOK_RE = re.compile(r'"((?:\\.|[^"\\])*)"', re.DOTALL)

# Adjacent C++ string literals are concatenated by the preprocessor.
# After concatenating, we still hold the raw source bytes which include
# escape sequences like \" \\ \n \t. Decode those into actual chars,
# then re-escape for Rust.
def cpp_str_decode(raw):
    out = []
    i = 0
    while i < len(raw):
        c = raw[i]
        if c == '\\' and i + 1 < len(raw):
            nxt = raw[i+1]
            if   nxt == '"':  out.append('"')
            elif nxt == '\\': out.append('\\')
            elif nxt == 'n':  out.append('\n')
            elif nxt == 't':  out.append('\t')
            elif nxt == 'r':  out.append('\r')
            elif nxt == '0':  out.append('\0')
            elif nxt == "'":  out.append("'")
            else:             out.append(nxt)
            i += 2
        else:
            out.append(c); i += 1
    return ''.join(out)

def coalesce_strings(arg):
    s = re.sub(r'/\*.*?\*/', '', arg, flags=re.DOTALL)
    s = re.sub(r'//[^\n]*', '', s)
    parts = STRING_TOK_RE.findall(s)
    leftover = STRING_TOK_RE.sub('', s).strip()
    if leftover:
        return None
    return ''.join(cpp_str_decode(p) for p in parts)

def rust_str(s):
    s = s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
    return f'"{s}"'

CONST_MAP_NUM = {
    'std::numeric_limits<Number>::epsilon()': 'f64::EPSILON',
    'std::numeric_limits<Number>::max()':     'f64::MAX',
    'std::numeric_limits<Number>::min()':     'f64::MIN_POSITIVE',
    'std::numeric_limits<Number>::infinity()':'f64::INFINITY',
    'std::numeric_limits<double>::epsilon()': 'f64::EPSILON',
    'std::numeric_limits<double>::max()':     'f64::MAX',
}
CONST_MAP_INT = {
    'std::numeric_limits<Index>::max()': 'i32::MAX',
    'std::numeric_limits<int>::max()':   'i32::MAX',
    'INT_MAX':                            'i32::MAX',
    'J_LAST_LEVEL':                       '13',
    'J_LAST_LEVEL - 1':                   '12',
    'J_ITERSUMMARY':                      '5',
    'MUMPS_INT_MAX':                      'i32::MAX',
    # MUMPS's USE_COMM_WORLD sentinel (defined in dmumps_c.h).
    'USE_COMM_WORLD':                    '-987654',
}

def rust_num(arg):
    """Translate a C++ numeric expression to a Rust f64 expression."""
    s = arg.strip()
    if s in CONST_MAP_NUM:
        return CONST_MAP_NUM[s]
    s2 = re.sub(r'/\*.*?\*/', '', s, flags=re.DOTALL)
    s2 = re.sub(r'//[^\n]*$', '', s2).strip()
    for k, v in CONST_MAP_NUM.items():
        s2 = s2.replace(k, v)
    # `std::pow(a, b)` or `pow(a, b)`  →  `(a).powf(b)`
    s2 = re.sub(r'\bstd\s*::\s*pow\s*\(([^,]+),\s*([^)]+)\)',
                lambda m: f'({m.group(1).strip()}).powf({m.group(2).strip()})',
                s2)
    s2 = re.sub(r'\bpow\s*\(([^,]+),\s*([^)]+)\)',
                lambda m: f'({m.group(1).strip()}).powf({m.group(2).strip()})',
                s2)
    # `1.` → `1.0` (digit + dot not followed by alphanumeric)
    s2 = re.sub(r'(?<![\w.])(\d+)\.(?![\w\d])', r'\1.0', s2)
    # Bare integer literal like `0`, `5000000` (no decimal/exponent/identifier)
    # in number context: append `.0`. We only do this when the entire (post-
    # whitespace, post-cast) expression is a plain integer literal -- we
    # don't try to rewrite arbitrary integer subexpressions.
    if re.fullmatch(r'-?\s*\d+', s2):
        s2 = s2 + '.0'
    return s2

def rust_int(arg):
    s = arg.strip()
    if s in CONST_MAP_INT:
        return CONST_MAP_INT[s]
    s2 = s
    for k, v in CONST_MAP_INT.items():
        s2 = s2.replace(k, v)
    return s2

def rust_bool(arg):
    s = arg.strip().lower()
    if s in ('true', '1'):  return 'true'
    if s in ('false', '0'): return 'false'
    raise ValueError(f"unexpected bool: {arg!r}")


# ---- Per-call dispatch ---------------------------------------------------

# For each Add* signature, give MIN_ARGS = required args without long_desc
# and without the trailing `advanced` flag.
SIG_MIN = {
    'AddBoolOption': 3,
    'AddNumberOption': 3,
    'AddIntegerOption': 3,
    'AddLowerBoundedNumberOption': 5,
    'AddBoundedNumberOption': 7,
    'AddLowerBoundedIntegerOption': 4,
    'AddBoundedIntegerOption': 5,
}

def parse_long_and_advanced(args, min_args):
    """args is the full arg list. Returns (long_desc_rust_str, advanced_bool_or_None)."""
    extras = len(args) - min_args
    if extras == 0:
        return rust_str(""), None
    if extras == 1:
        # Either `long_desc` (string) or `advanced` (bool). Disambiguate by literal form.
        last = args[-1].strip()
        if last in ('true', 'false'):
            return rust_str(""), last
        s = coalesce_strings(args[-1])
        if s is None:
            raise ValueError(f"can't classify trailing arg: {args[-1]!r}")
        return rust_str(s), None
    if extras == 2:
        s = coalesce_strings(args[-2])
        if s is None:
            raise ValueError(f"non-string long_desc: {args[-2]!r}")
        return rust_str(s), args[-1].strip()
    raise ValueError(f"too many trailing args ({extras}) for sig with min={min_args}: {args}")


def emit_call(rec):
    kind = rec['kind']
    args = rec['args']
    name_s = coalesce_strings(args[0])
    if name_s is None:
        raise ValueError(f"non-string name: {args[0]!r}")
    short_s = coalesce_strings(args[1])
    if short_s is None:
        raise ValueError(f"non-string short: {args[1]!r}")
    name = rust_str(name_s); short = rust_str(short_s)

    if kind in SIG_MIN:
        min_args = SIG_MIN[kind]
        long_d, _adv = parse_long_and_advanced(args, min_args)

        if kind == 'AddBoolOption':
            default = rust_bool(args[2])
            return f'    r.add_bool_option({name}, {short}, {default}, {long_d})?;'
        if kind == 'AddNumberOption':
            default = rust_num(args[2])
            return f'    r.add_number_option({name}, {short}, {default}, {long_d})?;'
        if kind == 'AddIntegerOption':
            default = rust_int(args[2])
            return f'    r.add_integer_option({name}, {short}, {default}, {long_d})?;'
        if kind == 'AddLowerBoundedNumberOption':
            lower = rust_num(args[2]); strict = rust_bool(args[3]); default = rust_num(args[4])
            return f'    r.add_lower_bounded_number_option({name}, {short}, {lower}, {strict}, {default}, {long_d})?;'
        if kind == 'AddBoundedNumberOption':
            lower = rust_num(args[2]); ls = rust_bool(args[3])
            upper = rust_num(args[4]); us = rust_bool(args[5])
            default = rust_num(args[6])
            return f'    r.add_bounded_number_option({name}, {short}, {lower}, {ls}, {upper}, {us}, {default}, {long_d})?;'
        if kind == 'AddLowerBoundedIntegerOption':
            lower = rust_int(args[2]); default = rust_int(args[3])
            return f'    r.add_lower_bounded_integer_option({name}, {short}, {lower}, {default}, {long_d})?;'
        if kind == 'AddBoundedIntegerOption':
            lower = rust_int(args[2]); upper = rust_int(args[3]); default = rust_int(args[4])
            return f'    r.add_bounded_integer_option({name}, {short}, {lower}, {upper}, {default}, {long_d})?;'

    m = re.match(r'AddStringOption(\d+)$', kind)
    if m:
        n = int(m.group(1))
        min_args = 3 + 2*n
        long_d, _adv = parse_long_and_advanced(args, min_args)
        default_s = coalesce_strings(args[2])
        if default_s is None: raise ValueError(f"non-string default: {args[2]!r}")
        default = rust_str(default_s)
        pairs = []
        base = 3
        for i in range(n):
            v = coalesce_strings(args[base + 2*i])
            d = coalesce_strings(args[base + 2*i + 1])
            if v is None: raise ValueError(f"non-string value: {args[base+2*i]!r}")
            if d is None: d = ""
            pairs.append(f'({rust_str(v)}, {rust_str(d)})')
        pairs_s = ', '.join(pairs)
        return f'    r.add_string_option({name}, {short}, {default}, &[{pairs_s}], {long_d})?;'

    raise ValueError(f"unhandled kind: {kind}")


# ---- Manually-curated additions for the dynamic AddStringOption(...) calls
# in IpAlgBuilder and IpTNLPAdapter, which build their option lists from
# build-time conditionals. We register the union of all possible values so
# upstream's option file format parses cleanly.
SPECIAL_INSERTS = {
    # (file, line) -> emitted Rust line
}


# ---- Driver --------------------------------------------------------------

def special_string_option(name, short, default, options_descs, long_d):
    pairs = ', '.join(f'({rust_str(v)}, {rust_str(d)})' for v, d in options_descs)
    return f'    r.add_string_option({rust_str(name)}, {rust_str(short)}, {rust_str(default)}, &[{pairs}], {rust_str(long_d)})?;'


def main():
    out = []
    out.append('//! Auto-generated upstream Ipopt option registrations.')
    out.append('//!')
    out.append('//! Walks each per-class `RegisterOptions` in the order specified by')
    out.append('//! `RegisterAllIpoptOptions`. Generated by `tools/optport/gen.py`;')
    out.append('//! re-run when bumping the upstream tag. The generated file should')
    out.append('//! not be edited manually.')
    out.append('//!')
    out.append('//! Most options register here just so the option-file parser accepts')
    out.append('//! the name; pounce strategies that consume an option pull it out of')
    out.append('//! the `OptionsList` at strategy-init time.')
    out.append('')
    out.append('#![allow(clippy::too_many_lines, clippy::approx_constant)]')
    out.append('')
    out.append('use pounce_common::exception::SolverException;')
    out.append('use pounce_common::reg_options::RegisteredOptions;')
    out.append('')
    out.append('pub fn register_all_upstream_options(r: &RegisteredOptions) -> Result<(), SolverException> {')

    extract_py = pathlib.Path(sys.argv[0]).parent / 'extract.py'

    for cpp_rel, leading_cat, label in CLASS_ORDER:
        cpp_path = ROOT / cpp_rel
        if not cpp_path.exists():
            out.append(f'    // SKIP missing file: {cpp_rel}')
            continue
        sub = subprocess.run(
            ['python3', str(extract_py), str(ROOT), cpp_rel],
            capture_output=True, text=True, check=True
        )
        recs = json.loads(sub.stdout)
        # filter empty-args (matched-in-comments) + filter SetRegisteringCategory
        # records that have empty value (parse glitch).
        recs = [r for r in recs
                if not (r['op'] == 'add' and len(r.get('args', [])) == 0)]
        if not recs:
            continue

        out.append('')
        out.append(f'    // ===== {label}::RegisterOptions ({cpp_rel}) =====')
        if leading_cat is not None:
            out.append(f'    r.set_registering_category({rust_str(leading_cat)});')

        for rec in recs:
            if rec['op'] == 'cat':
                cat_s = coalesce_strings(rec['value'])
                if cat_s is None:
                    cat_s = ""
                out.append(f'    r.set_registering_category({rust_str(cat_s)});')
            else:
                try:
                    out.append(emit_call(rec))
                except Exception as e:
                    name_arg = rec['args'][0] if rec.get('args') else '?'
                    out.append(f'    // SKIP {rec["kind"]} {name_arg}: {e}')

        # Per-file manual additions for dynamic AddStringOption(...) calls.
        if cpp_rel == 'Interfaces/IpTNLPAdapter.cpp':
            out.append(special_string_option(
                'dependency_detector',
                'Indicates which linear solver should be used to detect linearly dependent equality constraints.',
                'none',
                [('none', "don't check; no extra work at beginning"),
                 ('mumps', 'use MUMPS'),
                 ('wsmp', 'use WSMP'),
                 ('ma28', 'use MA28')],
                'This is experimental and does not work well.',
            ))
        elif cpp_rel == 'Algorithm/IpAlgBuilder.cpp':
            out.append(special_string_option(
                'linear_solver',
                'Linear solver used for step computations.',
                'ma57',
                [('ma27', 'use the Harwell routine MA27'),
                 ('ma57', 'use the Harwell routine MA57'),
                 ('ma77', 'use the Harwell routine HSL_MA77'),
                 ('ma86', 'use the Harwell routine HSL_MA86'),
                 ('ma97', 'use the Harwell routine HSL_MA97'),
                 ('pardiso', 'use the Pardiso package from pardiso-project.org'),
                 ('pardisomkl', 'use the Pardiso package from Intel MKL'),
                 ('spral', 'use the SPRAL package'),
                 ('wsmp', 'use WSMP package'),
                 ('mumps', 'use MUMPS package'),
                 ('custom', 'use custom linear solver (expert use)'),
                 ('feral', 'use FERAL pure-Rust sparse symmetric solver (pounce extension)')],
                'Determines which linear algebra package is to be used for the solution of the augmented linear system (for obtaining the search directions).',
            ))
            out.append(special_string_option(
                'linear_system_scaling',
                'Method for scaling the linear system.',
                'none',
                [('none', 'no scaling will be performed'),
                 ('mc19', 'use the Harwell routine MC19'),
                 ('slack-based', 'use the slack values')],
                'Determines the method used to compute symmetric scaling factors for the augmented system (see also the "linear_scaling_on_demand" option). This scaling is independent of the NLP problem scaling.',
            ))
            out.append(special_string_option(
                'nlp_scaling_method',
                'Select the technique used for scaling the NLP.',
                'gradient-based',
                [('none', 'no problem scaling will be performed'),
                 ('user-scaling', 'scaling parameters will come from the user'),
                 ('gradient-based', 'scale the problem so the maximum gradient at the starting point is nlp_scaling_max_gradient'),
                 ('equilibration-based', 'scale the problem so that first derivatives are of order 1 at random points (uses Harwell routine MC19)')],
                'Selects the technique used for scaling the problem internally before it is solved. For user-scaling, the parameters come from the NLP.',
            ))

    out.append('')
    out.append('    Ok(())')
    out.append('}')
    out.append('')

    sys.stdout.write('\n'.join(out))


if __name__ == '__main__':
    main()
