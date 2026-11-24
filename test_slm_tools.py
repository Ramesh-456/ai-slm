import json
import os
import tempfile
import unittest

import generate_dataset as gd
import java_tools as jt
import train_lora as tl
import validate_dataset as vd

HERE = os.path.dirname(os.path.abspath(__file__))

SNIPPET = '''
public class T {
    @Tool(description = "Find things, e.g. \\"red\\" ones. "
            + "Second part (with parens) and a, comma.")
    public List<Foo> findThings(
            @ToolParam(description = "keyword (like: saree, kurta)", required = false) String keyword,
            @ToolParam(description = "max results") int limit,
            @ToolParam(description = "UUID of the thing") UUID thingId) {
        return null;
    }

    @Tool(description = "No params.")
    public String noParams() { return ""; }
}
'''


class JavaToolsTests(unittest.TestCase):
    def test_snippet(self):
        tools = jt.extract_tools(SNIPPET)
        self.assertEqual([t["function"]["name"] for t in tools], ["findThings", "noParams"])
        first = tools[0]["function"]
        self.assertEqual(first["description"], 'Find things, e.g. "red" ones. Second part (with parens) and a, comma.')
        props = first["parameters"]["properties"]
        self.assertEqual(props["keyword"], {"type": "string", "description": "keyword (like: saree, kurta)"})
        self.assertEqual(props["limit"]["type"], "integer")
        self.assertEqual(props["thingId"], {"type": "string", "format": "uuid", "description": "UUID of the thing"})
        self.assertEqual(first["parameters"]["required"], ["limit", "thingId"])
        self.assertEqual(tools[1]["function"]["parameters"]["properties"], {})

    def test_system_prompt_directive_and_policies(self):
        self.assertEqual(jt.extract_system_prompt('x SYSTEM_PROMPT = "a " + "b\\n";'), "a b\n")
        directive = ('new StringBuilder(" LANGUAGE: x " + "y"); sb.append(" If the message language is unclear, default to ").append(hint);'
                     ' case "ta" -> "Tamil";')
        base, prefix, hints = jt.extract_language_directive(directive)
        self.assertEqual(base, " LANGUAGE: x y")
        self.assertEqual(jt.build_system_prompt("P.", (base, prefix, hints), "ta"),
                         "P. LANGUAGE: x y If the message language is unclear, default to Tamil.")
        self.assertEqual(jt.build_system_prompt("P.", (base, prefix, hints), None), "P. LANGUAGE: x y")

    @unittest.skipUnless(os.path.exists(gd.DEFAULT_TOOLS_JAVA), "service sources not present")
    def test_real_sources(self):
        with open(gd.DEFAULT_TOOLS_JAVA, encoding="utf-8") as fh:
            src = fh.read()
        names = {t["function"]["name"] for t in jt.extract_tools(src)}
        for expected in ("searchProducts", "trackOrder", "draftReturnRequest", "getStorePolicyInfo", "addToCart"):
            self.assertIn(expected, names)
        self.assertIn("payment", jt.extract_policies(src))


@unittest.skipUnless(os.path.exists(gd.DEFAULT_TOOLS_JAVA), "service sources not present")
class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(dir=HERE)
        cls.out = cls.tmp.name
        gd.main(["--n", "400", "--seed", "7", "--out-dir", cls.out])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_rows_valid(self):
        code = vd.main([os.path.join(self.out, "train.jsonl"), os.path.join(self.out, "val.jsonl"),
                        "--tools", os.path.join(self.out, "tools_schema.json")])
        self.assertEqual(code, 0)

    def test_deterministic_for_seed(self):
        other = tempfile.TemporaryDirectory(dir=HERE)
        try:
            gd.main(["--n", "400", "--seed", "7", "--out-dir", other.name])
            with open(os.path.join(self.out, "train.jsonl"), "rb") as a, open(os.path.join(other.name, "train.jsonl"), "rb") as b:
                self.assertEqual(a.read(), b.read())
        finally:
            other.cleanup()

    def test_split_and_coverage(self):
        stats = json.load(open(os.path.join(self.out, "dataset_stats.json"), encoding="utf-8"))
        self.assertEqual(stats["train"] + stats["val"], 400)
        self.assertGreater(stats["no_tool_examples"], 0)
        self.assertEqual(set(stats["examples_per_lang"]), {"en", "ta", "hi", "tanglish"})

    def test_validator_rejects_bad_rows(self):
        schemas = vd.load_tools(os.path.join(self.out, "tools_schema.json"))
        head = "You are LoomNest's friendly shopping assistant."
        uid = "11111111-1111-4111-8111-111111111111"

        def row(args, name="trackOrder", user="track my order " + uid):
            return {"messages": [
                {"role": "system", "content": head + " ..."}, {"role": "user", "content": user},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": name, "arguments": args}}]},
                {"role": "tool", "tool_call_id": "c1", "name": name, "content": "[]"},
                {"role": "assistant", "content": "done"}]}

        self.assertEqual(vd.validate_row(row({"orderId": uid}), schemas, head)[0], [])
        self.assertTrue(vd.validate_row(row({"orderId": "22222222-2222-4222-8222-222222222222"}), schemas, head)[0])
        self.assertTrue(vd.validate_row(row({}), schemas, head)[0])
        self.assertTrue(vd.validate_row(row({"orderId": uid, "extra": 1}), schemas, head)[0])
        self.assertTrue(vd.validate_row(row({"orderId": uid}, name="noSuchTool"), schemas, head)[0])

    def test_train_dry_run(self):
        code = tl.main(["--dry-run", "--train-file", os.path.join(self.out, "train.jsonl"),
                        "--val-file", os.path.join(self.out, "val.jsonl"), "--tools", os.path.join(self.out, "tools_schema.json")])
        self.assertEqual(code, 0)


class MaskingTests(unittest.TestCase):
    def test_only_assistant_turns_are_trained(self):
        messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function",
                     "function": {"name": "getMyCart", "arguments": {}}}]},
                    {"role": "tool", "tool_call_id": "1", "name": "getMyCart", "content": "{\"items\":[]}"},
                    {"role": "assistant", "content": "Your cart is empty right now."}]
        text = tl.render_chatml_approx(tl.normalize_messages(messages), [])
        spans = tl.assistant_spans(text)
        self.assertEqual(len(spans), 2)
        trained = "".join(text[a:b] for a, b in spans)
        self.assertIn("<tool_call>", trained)
        self.assertIn("Your cart is empty right now.<|im_end|>", trained)
        for forbidden in ("SYS", "hello", "tool_response"):
            self.assertNotIn(forbidden, trained)

    def test_labels_from_offsets(self):
        labels = tl.labels_from_offsets([10, 11, 12, 13], [(0, 3), (3, 6), (6, 9), (9, 12)], [(6, 12)])
        self.assertEqual(labels, [-100, -100, 12, 13])

    def test_string_arguments_are_normalised(self):
        msg = {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function",
               "function": {"name": "x", "arguments": "{\"a\": 1}"}}]}
        self.assertEqual(tl.normalize_messages([msg])[0]["tool_calls"][0]["function"]["arguments"], {"a": 1})


if __name__ == "__main__":
    unittest.main()
