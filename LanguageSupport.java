package com.loom.nest.recommendation.service.impl;

import java.util.Locale;

/** Builds the language instruction appended to every chat system prompt (Tamil / Tanglish / Hindi / English). */
public final class LanguageSupport {

    private LanguageSupport() {
    }

    public static String directive(String uiLanguage) {
        String hint = switch (uiLanguage == null ? "" : uiLanguage.trim().toLowerCase(Locale.ROOT)) {
            case "ta" -> "Tamil";
            case "hi" -> "Hindi";
            case "en" -> "English";
            default -> null;
        };
        StringBuilder sb = new StringBuilder(" LANGUAGE: detect the language of the user's latest message and reply in "
                + "that same language and script. English -> English. Tamil script -> Tamil. Hindi (Devanagari) -> "
                + "Hindi. Tamil written in English letters (Tanglish, e.g. 'enna price', 'vanakkam') -> Tanglish. "
                + "Hindi in English letters (Hinglish) -> Hinglish. Keep product names, prices, order ids and "
                + "numbers exactly as they are.");
        if (hint != null) {
            sb.append(" If the message language is unclear, default to ").append(hint).append('.');
        }
        return sb.toString();
    }
}
