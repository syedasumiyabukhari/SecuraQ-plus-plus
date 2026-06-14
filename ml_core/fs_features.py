"""
fs_features.py — Single source of truth for FS feature extraction.
Imported by both train_fs_direct.py (training) and backend_api.py (inference).
Adding features here automatically applies to both sides.
"""

import re
import numpy as np
import pandas as pd

# ── Vocabulary ────────────────────────────────────────────────────────────────

DANGEROUS_FUNCS = {
    "printf":   "FUNC_PRINTF",
    "sprintf":  "FUNC_SPRINTF",
    "fprintf":  "FUNC_FPRINTF",
    "snprintf": "FUNC_SNPRINTF",
    "wprintf":  "FUNC_WPRINTF",
    "swprintf": "FUNC_SWPRINTF",
    "vprintf":  "FUNC_VPRINTF",
    "vsprintf": "FUNC_VSPRINTF",
    "vfprintf": "FUNC_VFPRINTF",
    "syslog":   "FUNC_SYSLOG",
    "scanf":    "FUNC_SCANF",
    "gets":     "FUNC_GETS",
    "fgets":    "FUNC_FGETS",
    "puts":     "FUNC_PUTS",
    "fputs":    "FUNC_FPUTS",
    "read":     "FUNC_READ",
    "recv":     "FUNC_RECV",
}

KEYWORDS = {
    "int", "char", "float", "double", "return",
    "if", "else", "for", "while", "void", "static",
    "const", "unsigned", "signed", "long", "short",
    "struct", "typedef", "NULL", "null", "true", "false",
    "size_t", "wchar_t",
}

_USER_SOURCES = {'FUNC_SCANF', 'FUNC_GETS', 'FUNC_FGETS', 'FUNC_READ', 'FUNC_RECV'}

# ── Tokeniser ─────────────────────────────────────────────────────────────────

def _normalize_raw(code: str) -> str:
    code = re.sub(r'//.*',       '',  code)
    code = re.sub(r'/\*.*?\*/', '',  code, flags=re.DOTALL)
    code = re.sub(r'\bVAR_\d+\b', 'VAR', code)
    code = re.sub(r'\bSTR_\d+\b', 'STR', code)
    code = re.sub(r'\bFUNC\b',    'FUNC_GENERIC', code)
    code = re.sub(r'"[^"]*"',     'STR', code)
    code = re.sub(r'\b\d+\b',     'NUM', code)
    return code


def tokenize_code(code: str) -> list:
    code = _normalize_raw(code)
    tokens = re.findall(
        r'[A-Za-z_]\w*|==|!=|<=|>=|[\+\-\*/=<>!&|]+|[\(\)\{\}\[\];,]',
        code
    )
    _kw_lower  = {k.lower() for k in KEYWORDS}
    _SENTINELS = {'STR', 'VAR', 'NUM', 'FUNC_GENERIC'}
    out = []
    for tok in tokens:
        lk = tok.lower()
        if lk in DANGEROUS_FUNCS:
            out.append(DANGEROUS_FUNCS[lk])
        elif tok in _SENTINELS:
            out.append(tok)
        elif lk in _kw_lower:
            out.append(tok.upper())
        elif re.fullmatch(r'[A-Za-z_]\w*', tok):
            out.append('VAR')
        else:
            out.append(tok)
    return out


def token_string(code: str) -> str:
    return " ".join(tokenize_code(code))


# ── Rule features (existing) ──────────────────────────────────────────────────

def rule_features(tokens: list) -> dict:
    r = {}

    # printf patterns
    r['unsafe_printf'] = sum(
        1 for i in range(len(tokens)-2)
        if tokens[i] == 'FUNC_PRINTF' and tokens[i+1] == '(' and tokens[i+2] == 'VAR'
    )
    r['safe_printf'] = sum(
        1 for i in range(len(tokens)-2)
        if tokens[i] == 'FUNC_PRINTF' and tokens[i+1] == '(' and tokens[i+2] == 'STR'
    )

    # fprintf patterns
    r['unsafe_fprintf'] = sum(
        1 for i in range(len(tokens)-4)
        if tokens[i] == 'FUNC_FPRINTF' and tokens[i+1] == '('
        and tokens[i+3] == ',' and tokens[i+4] == 'VAR'
    )
    r['safe_fprintf'] = sum(
        1 for i in range(len(tokens)-4)
        if tokens[i] == 'FUNC_FPRINTF' and tokens[i+1] == '('
        and tokens[i+3] == ',' and tokens[i+4] == 'STR'
    )

    # sprintf patterns
    r['unsafe_sprintf'] = sum(
        1 for i in range(len(tokens)-4)
        if tokens[i] == 'FUNC_SPRINTF' and tokens[i+1] == '('
        and tokens[i+3] == ',' and tokens[i+4] == 'VAR'
    )

    # syslog
    r['unsafe_syslog'] = sum(
        1 for i in range(len(tokens)-4)
        if tokens[i] == 'FUNC_SYSLOG' and tokens[i+1] == '('
        and tokens[i+3] == ',' and tokens[i+4] == 'VAR'
    )

    # ── Option 2: expanded rules ─────────────────────────────────────────────

    # wprintf patterns
    r['unsafe_wprintf'] = sum(
        1 for i in range(len(tokens)-2)
        if tokens[i] == 'FUNC_WPRINTF' and tokens[i+1] == '(' and tokens[i+2] == 'VAR'
    )
    r['safe_wprintf'] = sum(
        1 for i in range(len(tokens)-2)
        if tokens[i] == 'FUNC_WPRINTF' and tokens[i+1] == '(' and tokens[i+2] == 'STR'
    )

    # vprintf / vsprintf patterns
    r['unsafe_vprintf'] = sum(
        1 for i in range(len(tokens)-2)
        if tokens[i] == 'FUNC_VPRINTF' and tokens[i+1] == '(' and tokens[i+2] == 'VAR'
    )
    r['unsafe_vsprintf'] = sum(
        1 for i in range(len(tokens)-4)
        if tokens[i] == 'FUNC_VSPRINTF' and tokens[i+1] == '('
        and tokens[i+3] == ',' and tokens[i+4] == 'VAR'
    )

    # snprintf: format is 3rd arg — FUNC_SNPRINTF ( VAR , NUM , VAR/STR
    r['unsafe_snprintf'] = sum(
        1 for i in range(len(tokens)-6)
        if tokens[i] == 'FUNC_SNPRINTF' and tokens[i+1] == '('
        and tokens[i+3] == ',' and tokens[i+5] == ',' and tokens[i+6] == 'VAR'
    )
    r['safe_snprintf'] = sum(
        1 for i in range(len(tokens)-6)
        if tokens[i] == 'FUNC_SNPRINTF' and tokens[i+1] == '('
        and tokens[i+3] == ',' and tokens[i+5] == ',' and tokens[i+6] == 'STR'
    )

    # aggregates
    r['total_unsafe'] = (
        r['unsafe_printf'] + r['unsafe_fprintf'] + r['unsafe_sprintf'] +
        r['unsafe_syslog'] + r['unsafe_wprintf'] + r['unsafe_vprintf'] +
        r['unsafe_vsprintf'] + r['unsafe_snprintf']
    )
    r['total_safe'] = (
        r['safe_printf'] + r['safe_fprintf'] +
        r['safe_wprintf'] + r['safe_snprintf']
    )
    r['has_unsafe']    = int(r['total_unsafe'] > 0)
    r['has_safe_only'] = int(r['total_safe'] > 0 and r['total_unsafe'] == 0)
    r['mixed']         = int(r['total_safe'] > 0 and r['total_unsafe'] > 0)

    # input sources
    r['has_scanf']      = int('FUNC_SCANF' in tokens)
    r['has_gets']       = int('FUNC_GETS'  in tokens)
    r['has_fgets']      = int('FUNC_FGETS' in tokens)
    r['has_read']       = int('FUNC_READ'  in tokens)
    r['has_recv']       = int('FUNC_RECV'  in tokens)
    r['n_user_sources'] = (r['has_scanf'] + r['has_gets'] + r['has_fgets'] +
                           r['has_read']  + r['has_recv'])

    # token counts
    r['n_printf_tok']  = tokens.count('FUNC_PRINTF')
    r['n_fprintf_tok'] = tokens.count('FUNC_FPRINTF')
    r['n_sprintf_tok'] = tokens.count('FUNC_SPRINTF')
    r['n_snprintf_tok']= tokens.count('FUNC_SNPRINTF')
    r['n_syslog_tok']  = tokens.count('FUNC_SYSLOG')
    r['n_wprintf_tok'] = tokens.count('FUNC_WPRINTF')
    r['n_vprintf_tok'] = tokens.count('FUNC_VPRINTF')
    r['n_vsprintf_tok']= tokens.count('FUNC_VSPRINTF')
    r['total_fmt_tok'] = (r['n_printf_tok'] + r['n_fprintf_tok'] +
                          r['n_sprintf_tok'] + r['n_snprintf_tok'] +
                          r['n_syslog_tok']  + r['n_wprintf_tok'] +
                          r['n_vprintf_tok'] + r['n_vsprintf_tok'])

    denom = r['total_unsafe'] + r['total_safe']
    r['unsafe_ratio']  = r['total_unsafe'] / (denom + 1e-9)

    r['n_var_tokens']  = tokens.count('VAR')
    r['n_str_tokens']  = tokens.count('STR')
    r['str_var_ratio'] = r['n_str_tokens'] / (r['n_var_tokens'] + 1e-9)
    r['n_tokens']      = len(tokens)

    r['danger_score']  = (
        4.0 * r['unsafe_printf']  +
        3.5 * r['unsafe_fprintf'] +
        3.0 * r['unsafe_syslog']  +
        2.5 * r['unsafe_sprintf'] +
        2.0 * r['unsafe_wprintf'] +
        2.0 * r['unsafe_vprintf'] +
        2.0 * r['unsafe_vsprintf']+
        1.5 * r['unsafe_snprintf']+
        1.0 * r['n_user_sources']
    )

    return r


# ── Option 1: Context window features ────────────────────────────────────────

def context_window_features(tokens: list) -> dict:
    f = {}

    # VAR = STR anywhere in code — VAR was assigned a string literal, likely safe
    f['prior_str_assignment'] = sum(
        1 for i in range(len(tokens)-2)
        if tokens[i] == 'VAR' and tokens[i+1] == '=' and tokens[i+2] == 'STR'
    )

    # printf(STR, VAR, ...) — has format literal + separate args = safe pattern
    f['printf_fmt_plus_args'] = 0
    for i in range(len(tokens)-4):
        if tokens[i] == 'FUNC_PRINTF' and tokens[i+1] == '(':
            window = tokens[i+2:i+9]
            if 'STR' in window and 'VAR' in window:
                f['printf_fmt_plus_args'] += 1

    # User input source appears before printf(VAR) — dangerous chain
    source_pos  = [i for i, t in enumerate(tokens) if t in _USER_SOURCES]
    printf_pos  = [
        i for i in range(len(tokens)-2)
        if tokens[i] == 'FUNC_PRINTF' and tokens[i+1] == '(' and tokens[i+2] == 'VAR'
    ]
    f['user_input_to_printf'] = 0
    if source_pos and printf_pos:
        for sp in source_pos:
            for pp in printf_pos:
                if sp < pp:
                    f['user_input_to_printf'] += 1
                    break

    # VAR = STR immediately before a printf call (within 10 tokens)
    f['local_str_before_printf'] = 0
    for pp in printf_pos:
        window_back = tokens[max(0, pp-10):pp]
        if 'VAR' in window_back and '=' in window_back and 'STR' in window_back:
            f['local_str_before_printf'] += 1

    # printf inside a conditional — slightly more likely to be safe (guarded)
    f['printf_in_conditional'] = 0
    for i in range(len(tokens)-1):
        if tokens[i] in ('IF', 'WHILE', 'FOR'):
            window = tokens[i:i+20]
            if 'FUNC_PRINTF' in window:
                f['printf_in_conditional'] += 1

    # ratio of safe context signals to unsafe signals
    total_context = f['prior_str_assignment'] + f['printf_fmt_plus_args']
    unsafe_context = f['user_input_to_printf']
    f['context_safe_ratio'] = total_context / (total_context + unsafe_context + 1e-9)

    return f


# ── Regex features (structural) ───────────────────────────────────────────────

def regex_features(code: str) -> dict:
    f = {}
    f['n_if']         = len(re.findall(r'\bif\s*\(', code))
    f['n_for']        = len(re.findall(r'\bfor\s*\(', code))
    f['n_while']      = len(re.findall(r'\bwhile\s*\(', code))
    f['n_return']     = len(re.findall(r'\breturn\b', code))
    f['code_len']     = len(code)
    f['n_semicolons'] = code.count(';')
    f['brace_depth']  = code.count('{') - code.count('}')
    f['n_pointers']   = code.count('*')
    f['n_ampersand']  = code.count('&')
    f['has_argv']     = int(bool(re.search(r'\bargv\b', code)))
    f['has_cin']      = int(bool(re.search(r'\bcin\b',  code)))
    f['n_str_raw']    = len(re.findall(r'\bSTR_\d+', code))
    f['n_var_raw']    = len(re.findall(r'\bVAR_\d+', code))
    f['str_var_raw']  = f['n_str_raw'] / (f['n_var_raw'] + 1e-9)
    return f


# ── Combined feature builder ──────────────────────────────────────────────────

def build_manual_features(code: str) -> dict:
    toks = tokenize_code(code)
    rf   = rule_features(toks)
    cx   = context_window_features(toks)
    rx   = regex_features(code)
    return {**rf, **cx, **rx}


def build_manual_matrix(df):
    rows = [build_manual_features(row['code']) for _, row in df.iterrows()]
    feat_df = pd.DataFrame(rows)
    X = feat_df.values.astype(np.float32)
    y = df['label'].values.astype(np.float32)
    return X, y, list(feat_df.columns)
