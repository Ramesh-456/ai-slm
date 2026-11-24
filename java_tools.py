"""Regex/scanner based extraction of the real tool schemas and prompts from the recommendation-service sources.

No Java parser is needed: the Spring AI annotations use plain string literals joined with '+', so a small
string-aware scanner is enough. Everything here is standard library only.
"""
import json
import re

_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "'": "'"}
_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')

JAVA_TO_SCHEMA = {
    "UUID": {"type": "string", "format": "uuid"},
    "String": {"type": "string"},
    "int": {"type": "integer"},
    "Integer": {"type": "integer"},
    "long": {"type": "integer"},
    "Long": {"type": "integer"},
    "BigDecimal": {"type": "number"},
    "double": {"type": "number"},
    "Double": {"type": "number"},
    "float": {"type": "number"},
    "boolean": {"type": "boolean"},
    "Boolean": {"type": "boolean"},
}

# Policy topics as the tool exposes them (the @ToolParam description lists these words).
POLICY_TOPICS = {"pay": "payment", "cancel": "cancellation", "return": "returns", "refund": "refund", "track": "tracking"}


def clean(text):
    """Normalises typographic characters so prompts stay plain and diff-friendly."""
    return text.replace("\u2014", "-").replace("\u2013", "-").replace("\u2019", "'")


def parse_string_literal(src, i):
    if src[i] != '"':
        raise ValueError("expected string literal at %d" % i)
    i += 1
    out = []
    while i < len(src):
        ch = src[i]
        if ch == "\\":
            nxt = src[i + 1]
            if nxt == "u":
                out.append(chr(int(src[i + 2:i + 6], 16)))
                i += 6
                continue
            out.append(_ESCAPES.get(nxt, nxt))
            i += 2
            continue
        if ch == '"':
            return "".join(out), i + 1
        out.append(ch)
        i += 1
    raise ValueError("unterminated string literal")


def parse_concat(src, i):
    """Parses "a" + "b" + ... starting at i; returns (joined_text, index_after_last_literal)."""
    parts = []
    end = i
    while True:
        while i < len(src) and src[i].isspace():
            i += 1
        if i >= len(src) or src[i] != '"':
            break
        text, i = parse_string_literal(src, i)
        parts.append(text)
        end = i
        j = i
        while j < len(src) and src[j].isspace():
            j += 1
        if j < len(src) and src[j] == "+":
            i = j + 1
            continue
        break
    if not parts:
        raise ValueError("no string literal at %d" % i)
    return "".join(parts), end


def find_matching_paren(src, open_idx):
    depth = 0
    i = open_idx
    while i < len(src):
        ch = src[i]
        if ch == '"':
            _, i = parse_string_literal(src, i)
            continue
        if ch == "'":
            i += 3 if src[i + 1] != "\\" else 4
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unbalanced parentheses")


def split_top_level(text):
    parts, depth, current, i = [], 0, [], 0
    while i < len(text):
        ch = text[i]
        if ch == '"':
            _, end = parse_string_literal(text, i)
            current.append(text[i:end])
            i = end
            continue
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
        i += 1
    if "".join(current).strip():
        parts.append("".join(current))
    return parts


def parse_param(text):
    text = text.strip()
    description, required = "", True
    if text.startswith("@ToolParam"):
        open_idx = text.index("(")
        close = find_matching_paren(text, open_idx)
        body = text[open_idx + 1:close]
        match = re.search(r"description\s*=\s*", body)
        if match:
            description, _ = parse_concat(body, match.end())
        if re.search(r"required\s*=\s*false", _STRING_RE.sub('""', body)):
            required = False
        text = text[close + 1:].strip()
    text = re.sub(r"^final\s+", "", text)
    java_type, name = text.rsplit(None, 1)
    schema = dict(JAVA_TO_SCHEMA.get(java_type.strip(), {"type": "string"}))
    if description:
        schema["description"] = clean(description)
    return {"name": name, "java_type": java_type.strip(), "schema": schema, "required": required}


def extract_tools(src):
    """Returns OpenAI-style tool definitions, in source order, for every @Tool method in src."""
    tools = []
    for match in re.finditer(r"@Tool\s*\(", src):
        open_idx = match.end() - 1
        close = find_matching_paren(src, open_idx)
        body = src[open_idx + 1:close]
        desc_match = re.search(r"description\s*=\s*", body)
        if not desc_match:
            continue
        description, _ = parse_concat(body, desc_match.end())
        sig = re.compile(r"public\s+[^()=;{]+?\s+(\w+)\s*\(").search(src, close)
        if not sig or sig.start() - close > 300:
            raise ValueError("no method signature after @Tool at offset %d" % match.start())
        popen = sig.end() - 1
        pclose = find_matching_paren(src, popen)
        params = [parse_param(p) for p in split_top_level(src[popen + 1:pclose]) if p.strip()]
        tools.append({
            "type": "function",
            "function": {
                "name": sig.group(1),
                "description": clean(description),
                "parameters": {
                    "type": "object",
                    "properties": {p["name"]: p["schema"] for p in params},
                    "required": [p["name"] for p in params if p["required"]],
                },
            },
        })
    return tools


def extract_system_prompt(src):
    match = re.search(r"SYSTEM_PROMPT\s*=\s*", src)
    if not match:
        raise ValueError("SYSTEM_PROMPT not found")
    return clean(parse_concat(src, match.end())[0])


def extract_language_directive(src):
    """Returns (base_directive, unclear_sentence_prefix, {ui_code: language_name}) from LanguageSupport."""
    match = re.search(r"new\s+StringBuilder\s*\(\s*", src)
    if not match:
        raise ValueError("StringBuilder directive not found in LanguageSupport")
    base = clean(parse_concat(src, match.end())[0])
    unclear = re.search(r'\.append\(\s*(" If the message language is unclear, default to ")\s*\)', src)
    prefix = parse_string_literal(unclear.group(1), 0)[0] if unclear else " If the message language is unclear, default to "
    hints = dict(re.findall(r'case\s+"(\w+)"\s*->\s*"(\w+)"', src))
    return base, prefix, hints


def extract_policies(src):
    """Maps topic (payment/cancellation/returns/refund/tracking/general) to the canned policy text."""
    method = re.search(r"static\s+String\s+storePolicyText\s*\(", src) or re.search(r"String\s+getStorePolicyInfo\s*\(", src)
    if not method:
        raise ValueError("store policy text not found")
    body = src[method.start():]
    policies = {}
    for key, topic in POLICY_TOPICS.items():
        found = re.search(r'key\.contains\("%s"\)[^{]*\{\s*return\s+' % key, body)
        if found:
            policies[topic] = clean(parse_concat(body, found.end())[0])
    general = re.search(r"\}\s*return\s+(\")", body[body.find("key.contains"):])
    if general:
        start = body.find("key.contains")
        policies["general"] = clean(parse_concat(body[start:], general.start(1))[0])
    if not policies:
        raise ValueError("could not parse any policy text")
    return policies


def build_system_prompt(prompt, directive, ui_language):
    """Mirrors ChatbotServiceImpl: SYSTEM_PROMPT + LanguageSupport.directive(language)."""
    base, unclear_prefix, hints = directive
    text = prompt + base
    hint = hints.get((ui_language or "").strip().lower())
    if hint:
        text += unclear_prefix + hint + "."
    return text


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
