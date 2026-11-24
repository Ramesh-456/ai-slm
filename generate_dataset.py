#!/usr/bin/env python3
"""Builds a synthetic tool-calling SFT dataset for the LoomNest shopping assistant (standard library only).

Tool schemas, the system prompt, the language directive and the canned policy texts are read from the real
recommendation-service sources, so the dataset follows the code. Output is ChatML/OpenAI-style chat JSONL:
{"messages": [system, user, assistant(tool_calls), tool, assistant, ...], "meta": {...}}.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys
import uuid

import java_tools as jt
import slm_lexicon as lx

HERE = os.path.dirname(os.path.abspath(__file__))
JAVA_BASE = os.path.join(HERE, "..", "recommendation-service", "src", "main", "java", "com", "loom", "nest", "recommendation")
DEFAULT_TOOLS_JAVA = os.path.join(JAVA_BASE, "ai", "ShoppingAssistantTools.java")
DEFAULT_CHATBOT_JAVA = os.path.join(JAVA_BASE, "service", "impl", "ChatbotServiceImpl.java")
DEFAULT_LANGUAGE_JAVA = os.path.join(JAVA_BASE, "service", "impl", "LanguageSupport.java")

YES_CONNECT = {"en": ["yes please", "yes, connect me"], "ta": ["ஆம், செய்யுங்கள்"], "hi": ["हाँ, कर दो"], "tg": ["aama, pannunga"]}
QUESTION_DIRECT = {
    "en": "about product {id}: {q}", "ta": "{id} பொருளைப் பற்றி: {q}",
    "hi": "प्रोडक्ट {id} के बारे में: {q}", "tg": "product {id} pathi: {q}",
}
REVIEW_COMMENTS = [
    "Great quality, totally worth the price.", "Good product but delivery was a bit slow.",
    "Fits well and looks exactly like the photos.", "Decent for the price, packaging could be better.",
    "Excellent value for money, will buy again.",
]
REVIEWERS = ["Anita", "Ravi", "Priya", "Karthik", "Meera", "Suresh", "Divya"]
ORDER_STATUSES = ["PLACED", "CONFIRMED", "SHIPPED", "DELIVERED"]
CARRIERS = ["BlueDart", "Delhivery", "DTDC", "Ecom Express"]
NOTIFICATIONS = [
    ("Order shipped", "Your order is on its way."), ("Flash sale", "Up to 40% off on footwear today."),
    ("Refund processed", "Your refund has been credited."), ("Order delivered", "Your order was delivered."),
]
SMALL_SEARCH_LIMIT = 5
FILLER_PREFIX = {
    "en": ["hey, ", "hi, ", "please ", "quick question: ", "hello, ", "ok so "],
    "ta": ["ஹலோ, ", "தயவுசெய்து ", "ஒரு சின்ன கேள்வி: ", "சரி, "],
    "hi": ["अरे, ", "कृपया ", "सुनो, ", "एक सवाल: ", "ठीक है, "],
    "tg": ["hey, ", "bro, ", "please ", "sollunga, ", "ok, "],
}
FILLER_SUFFIX = {
    "en": [" please", " thanks", " asap", " pls", " :)"],
    "ta": [" தயவுசெய்து", " நன்றி", " சீக்கிரம்"],
    "hi": [" कृपया", " धन्यवाद", " जल्दी"],
    "tg": [" please", " thanks", " seekiram", " pls"],
}


# ----------------------------------------------------------------------------------------------------------------------
# Catalogue
# ----------------------------------------------------------------------------------------------------------------------

def make_product(rng, name, category, price, mrp, keyword, pid=None):
    return {"id": pid or str(uuid.uuid5(uuid.NAMESPACE_URL, "loomnest/" + name)), "name": name,
            "description": "%s - good quality %s item." % (name, category.lower()), "price": price,
            "stock": rng.randint(5, 120), "category": {"name": category}, "mrp": mrp, "status": "APPROVED",
            "_kw": keyword}


def build_catalogue(rng, path):
    """Built-in sample catalogue, or a JSON export (list of {id,name,price,mrp?,category:{name}|str})."""
    if not path:
        return [make_product(rng, *row) for row in lx.SAMPLE_CATALOGUE], dict(lx.TERMS)
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    raw = raw.get("content", raw) if isinstance(raw, dict) else raw
    terms = dict(lx.TERMS)
    items = []
    for entry in raw:
        category = entry.get("category")
        category = category.get("name") if isinstance(category, dict) else (category or "General")
        price = entry.get("price") or 0
        price = int(price) if float(price).is_integer() else float(price)
        keyword = category.lower()
        items.append(make_product(rng, entry["name"], category, price, entry.get("mrp") or price, keyword, entry.get("id")))
        terms.setdefault(keyword, {lang: [keyword] for lang in lx.LANGS})
    if not items:
        raise SystemExit("catalogue %s is empty" % path)
    return items, terms


def public(product):
    return {k: v for k, v in product.items() if not k.startswith("_")}


def catalogue_search(catalogue, keyword, max_price=None, limit=SMALL_SEARCH_LIMIT):
    key = (keyword or "").lower()
    hits = [p for p in catalogue if (not key or key in (p["name"] + " " + p["category"]["name"]).lower()
                                     or key == p["_kw"]) and (max_price is None or p["price"] <= max_price)]
    return hits[:limit]


def nice_number(value):
    step = 50 if value < 1000 else 100 if value < 10000 else 500 if value < 50000 else 1000
    return int(-(-value // step) * step)


def money(value):
    return str(int(value)) if float(value).is_integer() else "%.2f" % value


def typo(rng, word):
    if len(word) < 4 or " " in word:
        return word
    i = rng.randint(1, len(word) - 2)
    op = rng.choice(("drop", "swap", "dup"))
    if op == "drop":
        return word[:i] + word[i + 1:]
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:] if i + 2 <= len(word) else word
    return word[:i] + word[i] + word[i:]


# ----------------------------------------------------------------------------------------------------------------------
# Example generator
# ----------------------------------------------------------------------------------------------------------------------

class Generator:
    def __init__(self, rng, catalogue, terms, policies, args_as_string=False):
        self.rng = rng
        self.catalogue = catalogue
        self.terms = terms
        self.policies = policies
        self.args_as_string = args_as_string
        self.lang = "en"
        self.tools_used = []
        self.messages = []

    # -- plumbing ------------------------------------------------------------------------------------------------
    def uid(self):
        return str(uuid.UUID(int=self.rng.getrandbits(128), version=4))

    def user(self, text):
        self.messages.append({"role": "user", "content": text})

    def say(self, text):
        self.messages.append({"role": "assistant", "content": text})

    def call(self, name, args, result):
        call_id = "call_%08x" % self.rng.getrandbits(32)
        arguments = jt.dumps(args) if self.args_as_string else args
        self.messages.append({"role": "assistant", "content": "",
                              "tool_calls": [{"id": call_id, "type": "function",
                                              "function": {"name": name, "arguments": arguments}}]})
        content = result if isinstance(result, str) else jt.dumps(result)
        self.messages.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": content})
        self.tools_used.append(name)

    def pack(self, key, **kw):
        template = lx.PACK[self.lang][key]
        if isinstance(template, list):
            template = self.rng.choice(template)
        return template.format(**kw)

    def utt(self, intent, **kw):
        text = self.rng.choice(lx.UTT[intent][self.lang]).format(**kw)
        if intent not in ("confirm",) and self.rng.random() < 0.45:
            if self.rng.random() < 0.6:
                text = self.rng.choice(FILLER_PREFIX[self.lang]) + text
            else:
                text = text + self.rng.choice(FILLER_SUFFIX[self.lang])
        return text

    def product_lines(self, products):
        return "\n".join("%d. %s - \u20b9%s" % (i, p["name"], money(p["price"])) for i, p in enumerate(products, 1))

    def pick_term(self, allow_unknown=True):
        if allow_unknown and self.rng.random() < 0.07:
            word = self.rng.choice(lx.UNKNOWN_TERMS)
            return word.lower(), word
        canon = self.rng.choice(sorted(self.terms))
        forms = self.terms[canon].get(self.lang) or [canon]
        form = self.rng.choice(forms)
        if self.lang in ("en", "tg") and self.rng.random() < 0.15:
            form = typo(self.rng, form)
        return canon, form

    def budget_for(self, canon):
        pool = [p["price"] for p in self.catalogue if p["_kw"] == canon] or [p["price"] for p in self.catalogue]
        return nice_number(self.rng.choice(pool) * self.rng.choice((1.0, 1.2, 1.5, 2.0)))

    def random_product(self):
        return self.rng.choice(self.catalogue)

    # -- scenarios -----------------------------------------------------------------------------------------------
    def search(self):
        canon, form = self.pick_term()
        with_budget = self.rng.random() < 0.55
        args = {"keyword": canon}
        if with_budget:
            budget = self.budget_for(canon)
            args["maxPrice"] = budget
            self.user(self.utt("search_budget", term=form, n=budget))
        else:
            self.user(self.utt("search_plain", term=form))
        results = catalogue_search(self.catalogue, canon, args.get("maxPrice"))
        self.call("searchProducts", args, [public(p) for p in results])
        self.say(self.pack("found", lines=self.product_lines(results)) if results else self.pack("none"))
        if len(results) >= 2 and self.rng.random() < 0.40:
            self.follow_up(results, canon)
            return "search_followup"
        return "search"

    def follow_up(self, products, keyword):
        kinds = ["similar", "fbt", "sentiment", "revsum", "reviews", "question", "cart_add", "wl_add", "compare", "cheaper"]
        weights = [14, 10, 12, 10, 8, 10, 10, 8, 10, 8]
        kind = self.rng.choices(kinds, weights)[0]
        index = self.rng.randrange(min(3, len(products)))
        ref = products[index]
        ordinal = lx.ORDINALS[self.lang][index]
        if kind == "similar":
            self.user(self.utt("similar_fu", ord=ordinal))
            others = [p for p in self.catalogue if p["category"]["name"] == ref["category"]["name"] and p["id"] != ref["id"]][:5]
            self.call("getSimilarProducts", {"productId": ref["id"], "limit": 5}, [public(p) for p in others])
            self.say(self.pack("similar", lines=self.product_lines(others)) if others else self.pack("similar_none"))
        elif kind == "fbt":
            self.user(self.utt("fbt_fu", ord=ordinal))
            self.fbt(ref)
        elif kind == "sentiment":
            self.user(self.utt("sentiment_fu", ord=ordinal))
            self.sentiment(ref)
        elif kind == "revsum":
            self.user(self.utt("revsum_fu", ord=ordinal))
            self.review_summary(ref)
        elif kind == "reviews":
            self.user(self.utt("reviews_fu", ord=ordinal))
            self.reviews(ref)
        elif kind == "question":
            question = self.rng.choice(lx.PRODUCT_QA)
            self.user(self.utt("question_fu", ord=ordinal, q=question[0][self.lang]))
            self.product_question(ref, question)
        elif kind == "cart_add":
            self.user(self.utt("cart_add_fu", ord=ordinal))
            self.call("addToCart", {"productId": ref["id"], "quantity": 1}, "Added to your cart!")
            self.say(self.pack("cart_added"))
        elif kind == "wl_add":
            self.user(self.utt("wl_add_fu", ord=ordinal))
            self.call("addToWishlist", {"productId": ref["id"]}, "Saved to your wishlist!")
            self.say(self.pack("wl_added"))
        elif kind == "compare":
            self.user(self.utt("compare_fu"))
            a, b = products[0], products[1]
            cheap, diff = (a, b["price"] - a["price"]) if a["price"] <= b["price"] else (b, a["price"] - b["price"])
            self.say(self.pack("compare", a=a["name"], pa=money(a["price"]), b=b["name"], pb=money(b["price"]),
                               cheap=cheap["name"], diff=money(diff)))
        else:
            prices = sorted(p["price"] for p in products)
            ceiling = nice_number(prices[len(prices) // 2]) if len(prices) > 2 else nice_number(prices[0] + 1)
            self.user(self.utt("cheaper_fu", n=ceiling))
            results = catalogue_search(self.catalogue, keyword, ceiling)
            self.call("searchProducts", {"keyword": keyword, "maxPrice": ceiling}, [public(p) for p in results])
            self.say(self.pack("found", lines=self.product_lines(results)) if results else self.pack("none"))

    def fbt(self, ref):
        pool = [p for p in self.catalogue if p["category"]["name"] != ref["category"]["name"]]
        picks = self.rng.sample(pool, k=min(3, len(pool))) if pool and self.rng.random() > 0.1 else []
        self.call("getFrequentlyBoughtTogether", {"productId": ref["id"], "limit": 4}, [public(p) for p in picks])
        self.say(self.pack("fbt", lines=self.product_lines(picks)) if picks else self.pack("fbt_none"))

    def sentiment(self, ref):
        total = self.rng.choice((0, 0, 8, 15, 32, 60)) if self.rng.random() < 0.15 else self.rng.choice((8, 15, 32, 60, 120))
        pos = int(total * self.rng.uniform(0.5, 0.85))
        neg = int((total - pos) * self.rng.uniform(0.1, 0.6))
        neu = total - pos - neg
        avg = round(self.rng.uniform(3.4, 4.7), 1) if total else 0.0
        overall = "NO_DATA" if not total else ("POSITIVE" if pos / total > 0.6 else "MIXED")
        result = {"productId": ref["id"], "totalReviews": total, "positiveCount": pos, "neutralCount": neu,
                  "negativeCount": neg, "averageRating": avg, "overallSentiment": overall,
                  "summary": "No reviews yet." if not total else "Most reviewers are happy with this product."}
        self.call("getProductSentiment", {"productId": ref["id"]}, result)
        if total:
            share = lambda n: int(round(100.0 * n / total))
            self.say(self.pack("sentiment", avg=avg, pos=share(pos), neu=share(neu), neg=share(neg)))
        else:
            self.say(self.pack("sentiment_none"))

    def review_summary(self, ref):
        pro, con = self.rng.choice(lx.ASPECTS_PRO), self.rng.choice(lx.ASPECTS_CON)
        summary = "Reviewers like the %s, though some mention %s." % (pro["en"], con["en"])
        self.call("getProductReviewSummary", {"productId": ref["id"]},
                  {"productId": ref["id"], "reviewCount": self.rng.choice((6, 12, 25, 48)), "summary": summary})
        self.say(self.pack("revsum", pro=pro[self.lang], con=con[self.lang]))

    def reviews(self, ref):
        count = self.rng.choice((0, 2, 3, 3)) if self.rng.random() < 0.9 else 0
        items = []
        for comment in self.rng.sample(REVIEW_COMMENTS, k=count):
            items.append({"id": self.uid(), "productId": ref["id"], "username": self.rng.choice(REVIEWERS),
                          "rating": self.rng.randint(3, 5), "comment": comment, "createdAt": "2026-02-%02dT10:15:00" % self.rng.randint(1, 28)})
        self.call("getProductReviews", {"productId": ref["id"], "limit": 3}, items)
        if items:
            lines = "\n".join("\"%s\" - %s (%d/5)" % (r["comment"], r["username"], r["rating"]) for r in items)
            self.say(self.pack("reviews_intro", lines=lines))
        else:
            self.say(self.pack("reviews_none"))

    def product_question(self, ref, question):
        texts, grounded, answers, answerable = question
        self.call("askProductQuestion", {"productId": ref["id"], "question": texts[self.lang]},
                  {"productId": ref["id"], "answer": grounded, "answerable": answerable, "sources": ["product details"] if answerable else []})
        self.say(answers[self.lang] if answerable else self.pack("qa_unknown"))

    def trending(self):
        k = self.rng.choice((3, 4, 5, 6, 8))
        text = self.utt("trending", k=k)
        self.user(text)
        limit = k if str(k) in text else 5
        picks = self.rng.sample(self.catalogue, k=min(limit, len(self.catalogue)))
        self.call("getTrendingProducts", {"limit": limit}, [public(p) for p in picks])
        self.say(self.pack("trending", lines=self.product_lines(picks)))

    def personalized(self):
        self.user(self.utt("personalized"))
        picks = self.rng.sample(self.catalogue, k=5)
        self.call("getPersonalizedRecommendations", {"limit": 5}, [public(p) for p in picks])
        self.say(self.pack("personal", lines=self.product_lines(picks)))

    def orders(self):
        k = self.rng.choice((3, 5, 5, 10))
        text = self.utt("orders", k=k)
        self.user(text)
        limit = k if str(k) in text else 5
        count = 0 if self.rng.random() < 0.1 else self.rng.randint(1, min(limit, 4))
        items = [{"id": self.uid(), "userId": self.uid(), "orderDate": "2026-0%d-%02dT11:20:00" % (self.rng.randint(1, 5), self.rng.randint(1, 28)),
                  "total": self.random_product()["price"], "status": self.rng.choice(ORDER_STATUSES), "items": []} for _ in range(count)]
        self.call("getMyOrders", {"limit": limit}, items)
        if items:
            lines = "\n".join("%s - %s - \u20b9%s - %s" % (o["id"], o["orderDate"][:10], money(o["total"]), o["status"]) for o in items)
            self.say(self.pack("orders", lines=lines))
        else:
            self.say(self.pack("orders_none"))

    def cart_view(self):
        self.user(self.utt("cart"))
        picks = [] if self.rng.random() < 0.15 else self.rng.sample(self.catalogue, k=self.rng.randint(1, 3))
        items = [{"id": self.uid(), "productId": p["id"], "quantity": self.rng.randint(1, 2), "priceAtAddition": p["price"],
                  "product": public(p)} for p in picks]
        total = sum(i["quantity"] * i["priceAtAddition"] for i in items)
        self.call("getMyCart", {}, {"items": items, "total": total})
        if items:
            lines = "\n".join("%s x%d - \u20b9%s" % (i["product"]["name"], i["quantity"], money(i["priceAtAddition"])) for i in items)
            self.say(self.pack("cart", lines=lines, total=money(total)))
        else:
            self.say(self.pack("cart_empty"))

    def wishlist_view(self):
        self.user(self.utt("wishlist"))
        picks = [] if self.rng.random() < 0.15 else self.rng.sample(self.catalogue, k=self.rng.randint(1, 4))
        items = [{"id": self.uid(), "productId": p["id"], "addedAt": "2026-03-01T09:00:00", "product": public(p)} for p in picks]
        self.call("getMyWishlist", {}, {"items": items})
        if items:
            self.say(self.pack("wl", lines=self.product_lines(picks)))
        else:
            self.say(self.pack("wl_empty"))

    def notifications(self):
        k = self.rng.choice((3, 5, 5, 10))
        text = self.utt("notifications", k=k)
        self.user(text)
        limit = k if str(k) in text else 5
        picks = [] if self.rng.random() < 0.1 else self.rng.sample(NOTIFICATIONS, k=self.rng.randint(1, len(NOTIFICATIONS)))
        items = [{"id": self.uid(), "title": t, "message": m, "type": "ORDER", "read": self.rng.random() < 0.5,
                  "createdAt": "2026-03-0%dT08:30:00" % self.rng.randint(1, 9)} for t, m in picks]
        self.call("getMyNotifications", {"limit": limit}, items)
        if items:
            self.say(self.pack("notifs", lines="\n".join("%s: %s" % (i["title"], i["message"]) for i in items)))
        else:
            self.say(self.pack("notifs_none"))

    def unread(self):
        self.user(self.utt("unread"))
        n = self.rng.choice((0, 1, 2, 3, 5, 9))
        self.call("getUnreadNotificationCount", {}, str(n))
        self.say(self.pack("unread", n=n))

    def profile(self):
        self.user(self.utt("profile"))
        name = self.rng.choice(REVIEWERS).lower() + str(self.rng.randint(1, 99))
        email = "%s@example.com" % name
        self.call("getMyProfile", {}, {"username": name, "email": email})
        self.say(self.pack("profile", email=email, username=name))

    def policy(self):
        topic = self.rng.choice(sorted(lx.POLICY_UTT))
        self.user(self.rng.choice(lx.POLICY_UTT[topic][self.lang]))
        text = self.policies[topic]
        self.call("getStorePolicyInfo", {"topic": topic}, text)
        self.say(text if self.lang == "en" else lx.POLICY_REPLY[topic][self.lang])

    def track(self):
        order_id = self.uid()
        self.user(self.utt("track", id=order_id))
        count = 0 if self.rng.random() < 0.1 else self.rng.randint(1, 2)
        lines_data = []
        for _ in range(count):
            status = self.rng.choice(ORDER_STATUSES)
            shipped = status in ("SHIPPED", "DELIVERED")
            lines_data.append({"id": self.uid(), "orderId": order_id, "productId": self.random_product()["id"], "quantity": 1,
                               "lineTotal": self.random_product()["price"], "status": status,
                               "carrier": self.rng.choice(CARRIERS) if shipped else None,
                               "trackingNumber": "TRK%09d" % self.rng.randint(0, 10 ** 9 - 1) if shipped else None})
        self.call("trackOrder", {"orderId": order_id}, lines_data)
        if lines_data:
            lines = "\n".join("%d. %s%s" % (i, d["status"], " - %s (%s)" % (d["carrier"], d["trackingNumber"]) if d["carrier"] else "")
                              for i, d in enumerate(lines_data, 1))
            self.say(self.pack("track", lines=lines))
        else:
            self.say(self.pack("track_none"))

    def cancel(self):
        order_id = self.uid()
        self.user(self.utt("cancel", id=order_id))
        if self.rng.random() < 0.7:
            self.call("cancelOrder", {"orderId": order_id}, "Your order has been cancelled and any online payment will be refunded automatically.")
            self.say(self.pack("cancel_ok"))
        else:
            self.call("cancelOrder", {"orderId": order_id}, "That order can't be cancelled right now.")
            self.say(self.pack("cancel_fail"))

    def refund(self):
        order_id = self.uid()
        self.user(self.utt("refund", id=order_id))
        count = 0 if self.rng.random() < 0.2 else 1
        items = [{"id": self.uid(), "orderId": order_id, "userId": self.uid(), "amount": self.random_product()["price"], "currency": "INR",
                  "status": self.rng.choice(("INITIATED", "PROCESSING", "COMPLETED")), "customerMessage": None,
                  "createdAt": "2026-03-10T12:00:00"} for _ in range(count)]
        self.call("getRefundStatus", {"orderId": order_id}, items)
        if items:
            self.say(self.pack("refund", lines="\n".join("\u20b9%s - %s" % (money(i["amount"]), i["status"]) for i in items)))
        else:
            self.say(self.pack("refund_none"))

    def return_flow(self):
        order_id = self.uid()
        index = self.rng.randrange(3)
        reason = lx.REASONS[self.lang][index]
        self.user(self.utt("return_start", id=order_id, reason=reason))
        eligible = self.rng.random() < 0.8
        line_id, price = self.uid(), self.random_product()["price"]
        draft = {"orderId": order_id, "submitted": False,
                 "eligibleItems": [{"sellerOrderLineId": line_id, "productId": self.random_product()["id"], "quantity": 1, "lineTotal": price}] if eligible else [],
                 "notEligible": [] if eligible else ["Item %s is SHIPPED - returns open only after delivery." % self.uid()],
                 "proposedReason": reason,
                 "nextSteps": ["Confirm which item(s) to return and the reason with the shopper.",
                               "Call requestOrderReturn with each sellerOrderLineId and the reason."] if eligible else
                              ["Nothing in this order can be returned right now.", "If something is wrong with the order, offer to escalate to human support."]}
        self.call("draftReturnRequest", {"orderId": order_id, "reason": reason}, draft)
        if eligible:
            self.say(self.pack("draft", item="1 x \u20b9%s" % money(price), reason=reason))
            self.user(self.utt("confirm"))
            self.call("requestOrderReturn", {"sellerOrderLineId": line_id, "reason": reason},
                      "Your return request has been submitted and is waiting for the seller's approval.")
            self.say(self.pack("return_ok"))
        else:
            self.say(self.pack("draft_none"))
            self.user(self.rng.choice(YES_CONNECT[self.lang]))
            self.call("escalateToHumanSupport",
                      {"summary": "Customer wants to return order %s but no item is eligible yet (%s)." % (order_id, lx.REASONS_CANON[index]),
                       "orderId": order_id},
                      "I've passed this to our support team with the details - they'll get back to you soon.")
            self.say(self.pack("escalate_ok"))

    def human(self):
        self.user(self.utt("human"))
        self.call("escalateToHumanSupport", {"summary": "Customer asked to speak to a human agent."},
                  "I've passed this to our support team with the details - they'll get back to you soon.")
        self.say(self.pack("escalate_ok"))

    def direct_product(self, kind):
        ref = self.random_product()
        pid = ref["id"]
        if kind == "similar":
            self.user(self.utt("similar_direct", id=pid))
            others = [p for p in self.catalogue if p["category"]["name"] == ref["category"]["name"] and p["id"] != pid][:5]
            self.call("getSimilarProducts", {"productId": pid, "limit": 5}, [public(p) for p in others])
            self.say(self.pack("similar", lines=self.product_lines(others)) if others else self.pack("similar_none"))
        elif kind == "fbt":
            self.user(self.utt("fbt_direct", id=pid))
            self.fbt(ref)
        elif kind == "sentiment":
            self.user(self.utt("sentiment_direct", id=pid))
            self.sentiment(ref)
        elif kind == "revsum":
            self.user(self.utt("revsum_direct", id=pid))
            self.review_summary(ref)
        elif kind == "reviews":
            self.user(self.utt("reviews_direct", id=pid))
            self.reviews(ref)
        elif kind == "question":
            question = self.rng.choice(lx.PRODUCT_QA)
            self.user(QUESTION_DIRECT[self.lang].format(id=pid, q=question[0][self.lang]))
            self.product_question(ref, question)
        elif kind == "cart_add":
            qty = self.rng.choice((1, 2, 3))
            self.user(self.utt("cart_add_direct", id=pid, k=qty))
            self.call("addToCart", {"productId": pid, "quantity": qty}, "Added to your cart!")
            self.say(self.pack("cart_added"))
        else:
            self.user(self.utt("wl_add_direct", id=pid))
            self.call("addToWishlist", {"productId": pid}, "Saved to your wishlist!")
            self.say(self.pack("wl_added"))

    # -- no-tool scenarios ---------------------------------------------------------------------------------------
    def no_tool(self, kind):
        if kind == "greet":
            self.user(self.utt("greet"))
            self.say(self.pack("greet"))
        elif kind == "thanks":
            self.user(self.utt("thanks"))
            self.say(self.pack("thanks"))
        elif kind == "capability_limit":
            self.user(self.utt("capability_limit"))
            self.say(self.pack("limit"))
        elif kind == "track_noid":
            self.user(self.utt("track_noid"))
            self.say(self.pack("ask_order_id"))
        elif kind == "refuse_other":
            self.user(self.utt("refuse_other", uid=self.uid()))
            self.say(self.pack("refuse_other"))
        elif kind == "injection":
            self.user(self.utt("injection"))
            self.say(self.pack("refuse_inject"))
        else:
            self.user(self.utt("ask_tools"))
            self.say(self.pack("about"))


SCENARIOS = [
    ("search", 16, lambda g: g.search()),
    ("trending", 5, lambda g: g.trending()),
    ("personalized", 3, lambda g: g.personalized()),
    ("orders", 4, lambda g: g.orders()),
    ("cart_view", 3, lambda g: g.cart_view()),
    ("wishlist_view", 2, lambda g: g.wishlist_view()),
    ("notifications", 3, lambda g: g.notifications()),
    ("unread", 2, lambda g: g.unread()),
    ("profile", 2, lambda g: g.profile()),
    ("policy", 7, lambda g: g.policy()),
    ("track", 4, lambda g: g.track()),
    ("cancel", 3, lambda g: g.cancel()),
    ("refund", 3, lambda g: g.refund()),
    ("return_flow", 5, lambda g: g.return_flow()),
    ("human", 2, lambda g: g.human()),
    ("direct_similar", 2, lambda g: g.direct_product("similar")),
    ("direct_fbt", 2, lambda g: g.direct_product("fbt")),
    ("direct_sentiment", 2, lambda g: g.direct_product("sentiment")),
    ("direct_revsum", 2, lambda g: g.direct_product("revsum")),
    ("direct_reviews", 2, lambda g: g.direct_product("reviews")),
    ("direct_question", 3, lambda g: g.direct_product("question")),
    ("direct_cart_add", 2, lambda g: g.direct_product("cart_add")),
    ("direct_wl_add", 2, lambda g: g.direct_product("wl_add")),
    ("greet", 3, lambda g: g.no_tool("greet")),
    ("thanks", 2, lambda g: g.no_tool("thanks")),
    ("capability_limit", 4, lambda g: g.no_tool("capability_limit")),
    ("track_noid", 3, lambda g: g.no_tool("track_noid")),
    ("refuse_other", 4, lambda g: g.no_tool("refuse_other")),
    ("injection", 6, lambda g: g.no_tool("injection")),
    ("ask_tools", 2, lambda g: g.no_tool("ask_tools")),
]


def ui_language_hint(rng, lang):
    roll = rng.random()
    if roll < 0.40:
        return {"en": "en", "ta": "ta", "hi": "hi", "tg": None}[lang]
    if roll < 0.45:
        return rng.choice(["en", "ta", "hi"])  # deliberately mismatching UI hint: the reply must follow the message
    return None


def generate(n, seed, catalogue_path, prompt, directive, policies, args_as_string=False, embed_tools=None):
    rng = random.Random(seed)
    catalogue, terms = build_catalogue(rng, catalogue_path)
    weights = [w for _, w, _ in SCENARIOS]
    lang_keys = list(lx.LANG_WEIGHTS)
    lang_weights = [lx.LANG_WEIGHTS[k] for k in lang_keys]
    rows, seen, attempts = [], set(), 0
    while len(rows) < n and attempts < n * 30:
        attempts += 1
        name, _, run = rng.choices(SCENARIOS, weights)[0]
        gen = Generator(rng, catalogue, terms, policies, args_as_string)
        gen.lang = rng.choices(lang_keys, lang_weights)[0]
        refined = run(gen) or name
        if name == "search" and refined == "search_followup":
            name = "search_followup"
        hint = ui_language_hint(rng, gen.lang)
        messages = [{"role": "system", "content": jt.build_system_prompt(prompt, directive, hint)}] + gen.messages
        key = hashlib.sha1(json.dumps(messages[1:], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        row = {"messages": messages, "meta": {"kind": name, "lang": lx.LANG_NAMES[gen.lang], "tools": list(gen.tools_used)}}
        if embed_tools:
            row["tools"] = embed_tools
        rows.append(row)
    if len(rows) < n:
        print("warning: only %d unique examples produced (requested %d)" % (len(rows), n), file=sys.stderr)
    return rows


def stratified_split(rows, val_fraction, seed):
    rng = random.Random(seed + 1)
    by_kind = collections.defaultdict(list)
    for index, row in enumerate(rows):
        by_kind[row["meta"]["kind"]].append(index)
    val_idx = set()
    for kind, indices in sorted(by_kind.items()):
        rng.shuffle(indices)
        take = int(round(len(indices) * val_fraction))
        if len(indices) >= 5:
            take = max(1, take)
        val_idx.update(indices[:take])
    train = [r for i, r in enumerate(rows) if i not in val_idx]
    val = [r for i, r in enumerate(rows) if i in val_idx]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def compute_stats(rows, tool_names):
    per_tool = collections.Counter()
    examples_with_tool = collections.Counter()
    per_lang = collections.Counter(r["meta"]["lang"] for r in rows)
    per_kind = collections.Counter(r["meta"]["kind"] for r in rows)
    lang_tool = collections.defaultdict(collections.Counter)
    multi_step = no_tool = multi_turn = 0
    for row in rows:
        used = row["meta"]["tools"]
        per_tool.update(used)
        examples_with_tool.update(set(used))
        for t in set(used):
            lang_tool[row["meta"]["lang"]][t] += 1
        no_tool += 0 if used else 1
        multi_step += 1 if len(used) >= 2 else 0
        multi_turn += 1 if sum(1 for m in row["messages"] if m["role"] == "user") >= 2 else 0
    return {"examples": len(rows), "tool_calls_per_tool": {t: per_tool.get(t, 0) for t in tool_names},
            "examples_per_lang": dict(sorted(per_lang.items())), "examples_per_kind": dict(sorted(per_kind.items())),
            "no_tool_examples": no_tool, "multi_tool_examples": multi_step, "multi_turn_examples": multi_turn,
            "tools_without_examples": [t for t in tool_names if per_tool.get(t, 0) == 0]}


def print_stats(stats, train_n, val_n):
    print("\nexamples: %d (train %d / val %d)" % (stats["examples"], train_n, val_n))
    print("\ntool calls per tool:")
    width = max(len(t) for t in stats["tool_calls_per_tool"])
    for tool, count in sorted(stats["tool_calls_per_tool"].items(), key=lambda kv: -kv[1]):
        print("  %-*s %5d" % (width, tool, count))
    print("\nexamples per language: %s" % stats["examples_per_lang"])
    print("no-tool examples (small talk / refusals / clarifications): %d" % stats["no_tool_examples"])
    print("multi-turn examples: %d | multi-tool-call examples: %d" % (stats["multi_turn_examples"], stats["multi_tool_examples"]))
    print("\nexamples per scenario kind:")
    for kind, count in sorted(stats["examples_per_kind"].items(), key=lambda kv: -kv[1]):
        print("  %-18s %5d" % (kind, count))
    if stats["tools_without_examples"]:
        print("\nWARNING: no examples for tools: %s (add templates in generate_dataset.py)" % stats["tools_without_examples"])


def read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Generate the LoomNest shopping-assistant tool-calling SFT dataset.")
    p.add_argument("--n", type=int, default=3000, help="number of examples (default %(default)s)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--out-dir", default=os.path.join(HERE, "data"))
    p.add_argument("--catalogue", help="optional JSON export of products ([{id,name,price,mrp,category}] or a page with 'content')")
    p.add_argument("--tools-java", default=DEFAULT_TOOLS_JAVA)
    p.add_argument("--chatbot-java", default=DEFAULT_CHATBOT_JAVA)
    p.add_argument("--language-java", default=DEFAULT_LANGUAGE_JAVA)
    p.add_argument("--arguments-as-string", action="store_true",
                   help="emit tool_call arguments as JSON strings (OpenAI wire format) instead of objects (HF chat-template format)")
    p.add_argument("--embed-tools", action="store_true", help="also copy the tool schema into every row under 'tools'")
    args = p.parse_args(argv)

    tools_src = read(args.tools_java)
    tools = jt.extract_tools(tools_src)
    prompt = jt.extract_system_prompt(read(args.chatbot_java))
    directive = jt.extract_language_directive(read(args.language_java))
    policies = jt.extract_policies(tools_src)
    names = [t["function"]["name"] for t in tools]
    print("extracted %d tools from %s" % (len(tools), os.path.basename(args.tools_java)))

    rows = generate(args.n, args.seed, args.catalogue, prompt, directive, policies, args.arguments_as_string,
                    tools if args.embed_tools else None)
    train, val = stratified_split(rows, args.val_fraction, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    for filename, subset in (("train.jsonl", train), ("val.jsonl", val)):
        with open(os.path.join(args.out_dir, filename), "w", encoding="utf-8", newline="\n") as fh:
            for row in subset:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out_dir, "tools_schema.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(tools, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    with open(os.path.join(args.out_dir, "system_prompt.txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(jt.build_system_prompt(prompt, directive, None) + "\n")
    stats = compute_stats(rows, names)
    with open(os.path.join(args.out_dir, "dataset_stats.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(dict(stats, train=len(train), val=len(val), seed=args.seed), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print_stats(stats, len(train), len(val))
    print("\nwrote %s (train.jsonl, val.jsonl, tools_schema.json, system_prompt.txt, dataset_stats.json)" % args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
