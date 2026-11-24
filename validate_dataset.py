#!/usr/bin/env python3
"""Validates the generated tool-calling JSONL against tools_schema.json (standard library only).

Checks per line: valid JSON, message-role sequence, tool names exist, arguments match the schema (required keys,
no unknown keys, types, UUID format), UUID arguments are grounded in earlier messages (no invented ids), every tool
call is answered by a matching tool message, and the example ends with a non-empty assistant text.
"""
import argparse
import json
import os
import re
import sys

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
HERE = os.path.dirname(os.path.abspath(__file__))


def load_tools(path):
    with open(path, "r", encoding="utf-8") as fh:
        tools = json.load(fh)
    return {t["function"]["name"]: t["function"]["parameters"] for t in tools}


def check_value(name, schema, value):
    kind = schema.get("type")
    if kind == "string":
        if not isinstance(value, str) or not value.strip():
            return "%s must be a non-empty string" % name
        if schema.get("format") == "uuid" and not UUID_RE.match(value):
            return "%s is not a UUID: %r" % (name, value)
    elif kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return "%s must be an integer, got %r" % (name, value)
        if value < 0:
            return "%s must not be negative" % name
    elif kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            return "%s must be a non-negative number, got %r" % (name, value)
    elif kind == "boolean":
        if not isinstance(value, bool):
            return "%s must be boolean" % name
    return None


def check_arguments(tool, params, arguments):
    errors = []
    properties = params.get("properties", {})
    for key in params.get("required", []):
        if key not in arguments:
            errors.append("%s: missing required argument '%s'" % (tool, key))
    for key, value in arguments.items():
        if key not in properties:
            errors.append("%s: unknown argument '%s'" % (tool, key))
            continue
        problem = check_value("%s.%s" % (tool, key), properties[key], value)
        if problem:
            errors.append(problem)
    return errors


def validate_row(row, schemas, prompt_head):
    errors, used = [], []
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return ["messages must be a list with at least system, user, assistant"], used
    if messages[0].get("role") != "system" or not messages[0].get("content", "").startswith(prompt_head):
        errors.append("first message must be the LoomNest system prompt")
    pending = {}
    context = messages[0].get("content", "")
    for index, msg in enumerate(messages[1:], 1):
        role = msg.get("role")
        if role == "user":
            if pending:
                errors.append("msg %d: user turn while tool calls are unanswered" % index)
            if not isinstance(msg.get("content"), str) or not msg["content"].strip():
                errors.append("msg %d: empty user content" % index)
            context += "\n" + msg.get("content", "")
        elif role == "assistant":
            calls = msg.get("tool_calls")
            if calls:
                if msg.get("content"):
                    errors.append("msg %d: assistant tool-call message should have empty content" % index)
                for call in calls:
                    fn = call.get("function", {})
                    name = fn.get("name")
                    if name not in schemas:
                        errors.append("msg %d: unknown tool '%s'" % (index, name))
                        continue
                    arguments = fn.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except ValueError:
                            errors.append("msg %d: arguments are not valid JSON" % index)
                            continue
                    if not isinstance(arguments, dict):
                        errors.append("msg %d: arguments must be an object" % index)
                        continue
                    errors.extend("msg %d: %s" % (index, e) for e in check_arguments(name, schemas[name], arguments))
                    for key, value in arguments.items():
                        prop = schemas[name]["properties"].get(key, {})
                        if prop.get("format") == "uuid" and isinstance(value, str) and value not in context:
                            errors.append("msg %d: %s.%s id %s never appears earlier (invented id)" % (index, name, key, value))
                    pending[call.get("id")] = name
                    used.append(name)
            else:
                if pending:
                    errors.append("msg %d: assistant text while tool calls are unanswered" % index)
                if not isinstance(msg.get("content"), str) or not msg["content"].strip():
                    errors.append("msg %d: empty assistant content" % index)
                context += "\n" + msg.get("content", "")
        elif role == "tool":
            expected = pending.pop(msg.get("tool_call_id"), None)
            if expected is None:
                errors.append("msg %d: tool message without matching tool call" % index)
            elif msg.get("name") != expected:
                errors.append("msg %d: tool name %s != call %s" % (index, msg.get("name"), expected))
            if not isinstance(msg.get("content"), str):
                errors.append("msg %d: tool content must be a string" % index)
            context += "\n" + str(msg.get("content", ""))
        else:
            errors.append("msg %d: unexpected role %r" % (index, role))
    last = messages[-1]
    if last.get("role") != "assistant" or last.get("tool_calls") or not str(last.get("content", "")).strip():
        errors.append("example must end with a non-empty assistant text")
    if pending:
        errors.append("unanswered tool calls: %s" % sorted(pending.values()))
    meta_tools = row.get("meta", {}).get("tools")
    if meta_tools is not None and meta_tools != used:
        errors.append("meta.tools %s does not match tool calls %s" % (meta_tools, used))
    return errors, used


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="*", help="JSONL files (default data/train.jsonl data/val.jsonl)")
    p.add_argument("--tools", default=os.path.join(HERE, "data", "tools_schema.json"))
    p.add_argument("--max-errors", type=int, default=20, help="stop printing after this many bad lines")
    args = p.parse_args(argv)
    files = args.files or [os.path.join(HERE, "data", "train.jsonl"), os.path.join(HERE, "data", "val.jsonl")]
    schemas = load_tools(args.tools)
    prompt_head = "You are LoomNest's friendly shopping assistant."

    total = bad = 0
    per_tool = {name: 0 for name in schemas}
    for path in files:
        with open(path, "r", encoding="utf-8") as fh:
            for number, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                total += 1
                try:
                    row = json.loads(line)
                except ValueError as exc:
                    bad += 1
                    print("%s:%d invalid JSON: %s" % (os.path.basename(path), number, exc))
                    continue
                errors, used = validate_row(row, schemas, prompt_head)
                for name in used:
                    per_tool[name] = per_tool.get(name, 0) + 1
                if errors:
                    bad += 1
                    if bad <= args.max_errors:
                        print("%s:%d" % (os.path.basename(path), number))
                        for err in errors[:4]:
                            print("    - " + err)
    print("validated %d examples in %d file(s): %d valid, %d invalid" % (total, len(files), total - bad, bad))
    missing = [n for n, c in per_tool.items() if c == 0]
    if missing:
        print("note: tools with no examples: %s" % missing)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
