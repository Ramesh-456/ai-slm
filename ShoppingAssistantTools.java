package com.loom.nest.recommendation.ai;

import java.math.BigDecimal;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.function.Consumer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.web.server.ResponseStatusException;

import com.loom.nest.contracts.UserContactDTO;
import com.loom.nest.recommendation.dto.ReviewSummaryResponseDTO;
import com.loom.nest.recommendation.dto.SentimentResponseDTO;
import com.loom.nest.recommendation.dto.cart.CartItemCreateRequestDTO;
import com.loom.nest.recommendation.dto.cart.CartResponseDTO;
import com.loom.nest.recommendation.dto.notification.NotificationResponseDTO;
import com.loom.nest.recommendation.dto.order.OrderResponseDTO;
import com.loom.nest.recommendation.dto.product.ProductResponseDTO;
import com.loom.nest.recommendation.dto.product.ProductSearchPageDTO;
import com.loom.nest.recommendation.dto.refund.RefundResponseDTO;
import com.loom.nest.recommendation.dto.review.ReviewResponseDTO;
import com.loom.nest.recommendation.dto.seller.ReasonRequestDTO;
import com.loom.nest.recommendation.dto.seller.SellerOrderTrackingDTO;
import com.loom.nest.recommendation.dto.wishlist.WishlistItemCreateRequestDTO;
import com.loom.nest.recommendation.dto.wishlist.WishlistResponseDTO;
import com.loom.nest.recommendation.feign.CartClient;
import com.loom.nest.recommendation.feign.NotificationClient;
import com.loom.nest.recommendation.feign.OrderClient;
import com.loom.nest.recommendation.feign.ProductClient;
import com.loom.nest.recommendation.feign.RefundClient;
import com.loom.nest.recommendation.feign.SellerOrderClient;
import com.loom.nest.recommendation.feign.UserClient;
import com.loom.nest.recommendation.feign.WishlistClient;
import com.loom.nest.recommendation.qa.ProductQaResponseDTO;
import com.loom.nest.recommendation.qa.ProductQaService;
import com.loom.nest.recommendation.rag.PolicyKnowledgeService;
import com.loom.nest.recommendation.rag.RagDtos;
import com.loom.nest.recommendation.service.RecommendationService;
import com.loom.nest.recommendation.service.ReviewSummaryService;
import com.loom.nest.recommendation.service.SentimentService;
import com.loom.nest.recommendation.service.impl.InternalNotifier;
import com.loom.nest.recommendation.service.impl.ReviewCacheService;

/**
 * Spring AI {@code @Tool} methods the shopping-assistant LLM can call to look up
 * real catalogue/recommendation/sentiment data instead of hallucinating it.
 * <p>
 * One instance is created PER chat request (not a Spring bean) so the current
 * shopper's identity is bound at construction time from the server-side JWT —
 * never accepted as an LLM-supplied argument — closing off any prompt-injection
 * path to another user's personalized data.
 */
public class ShoppingAssistantTools {

    private static final Logger log = LoggerFactory.getLogger(ShoppingAssistantTools.class);
    private static final int MAX_ACCUMULATED = 8;

    private final ProductClient productClient;
    private final RecommendationService recommendationService;
    private final SentimentService sentimentService;
    private final ReviewSummaryService reviewSummaryService;
    private final ReviewCacheService reviewCache;
    private final OrderClient orderClient;
    private final SellerOrderClient sellerOrderClient;
    private final RefundClient refundClient;
    private final CartClient cartClient;
    private final WishlistClient wishlistClient;
    private final NotificationClient notificationClient;
    private final UserClient userClient;
    private final InternalNotifier internalNotifier;
    private final ProductQaService productQaService;
    private final PolicyKnowledgeService policyKnowledge;
    private final UUID currentUserId;
    private final String authHeader;
    private final Consumer<List<ProductResponseDTO>> onProductsResolved;
    private final Map<UUID, ProductResponseDTO> accumulated = new LinkedHashMap<>();

    public ShoppingAssistantTools(ProductClient productClient, RecommendationService recommendationService,
            SentimentService sentimentService, ReviewSummaryService reviewSummaryService, ReviewCacheService reviewCache,
            OrderClient orderClient, SellerOrderClient sellerOrderClient, RefundClient refundClient, CartClient cartClient,
            WishlistClient wishlistClient, NotificationClient notificationClient, UserClient userClient,
            InternalNotifier internalNotifier, ProductQaService productQaService, PolicyKnowledgeService policyKnowledge,
            UUID currentUserId, String authHeader, Consumer<List<ProductResponseDTO>> onProductsResolved) {
        this.productClient = productClient;
        this.recommendationService = recommendationService;
        this.sentimentService = sentimentService;
        this.reviewSummaryService = reviewSummaryService;
        this.reviewCache = reviewCache;
        this.orderClient = orderClient;
        this.sellerOrderClient = sellerOrderClient;
        this.refundClient = refundClient;
        this.cartClient = cartClient;
        this.wishlistClient = wishlistClient;
        this.notificationClient = notificationClient;
        this.userClient = userClient;
        this.internalNotifier = internalNotifier;
        this.productQaService = productQaService;
        this.policyKnowledge = policyKnowledge;
        this.currentUserId = currentUserId;
        this.authHeader = authHeader;
        this.onProductsResolved = onProductsResolved;
    }

    @Tool(description = "Search the product catalogue by keyword and/or a maximum price. Use this whenever "
            + "the shopper mentions a specific product, category, or budget.")
    public List<ProductResponseDTO> searchProducts(
            @ToolParam(description = "product keyword to search for", required = false) String keyword,
            @ToolParam(description = "maximum price filter in rupees", required = false) BigDecimal maxPrice) {
        try {
            log.info("Calling {} service -> searchProducts(keyword={}, maxPrice={})", "msvc-products", keyword, maxPrice);
            ProductSearchPageDTO page = productClient.searchProducts(keyword, maxPrice, 0, 5);
            List<ProductResponseDTO> results = page != null && page.getContent() != null ? page.getContent() : List.of();
            log.info("searchProducts tool returning {} product(s)", results.size());
            return publish(results);
        } catch (Exception ex) {
            log.warn("searchProducts tool failed: {}", ex.getMessage());
            return List.of();
        }
    }

    @Tool(description = "Get currently trending/popular/best-selling products across the store. Use this when "
            + "the shopper asks for trending or popular items without a specific keyword.")
    public List<ProductResponseDTO> getTrendingProducts(
            @ToolParam(description = "max number of products to return") int limit) {
        return publish(recommendationService.findTrendingProducts(limit));
    }

    @Tool(description = "Get personalized product recommendations for the CURRENT shopper, based on their own "
            + "order history. Use this when the shopper asks to be recommended something 'for me' with no "
            + "specific keyword. Falls back to trending products if nothing personalized is available.")
    public List<ProductResponseDTO> getPersonalizedRecommendations(
            @ToolParam(description = "max number of products to return") int limit) {
        if (currentUserId != null) {
            List<ProductResponseDTO> personalized = recommendationService.findForUser(currentUserId, authHeader, limit);
            if (!personalized.isEmpty()) {
                return publish(personalized);
            }
        }
        return publish(recommendationService.findTrendingProducts(limit));
    }

    @Tool(description = "Find products similar to a given reference product (same category/price-range/name "
            + "overlap). Use this when the shopper asks for something 'like' or 'similar to' a product.")
    public List<ProductResponseDTO> getSimilarProducts(
            @ToolParam(description = "UUID of the reference product") UUID productId,
            @ToolParam(description = "max number of products to return") int limit) {
        try {
            return publish(recommendationService.findSimilarProducts(productId, limit));
        } catch (Exception ex) {
            log.warn("getSimilarProducts tool failed: {}", ex.getMessage());
            return List.of();
        }
    }

    @Tool(description = "Find products frequently bought together with a given product. Use this when the "
            + "shopper asks what goes well with, or is commonly bought alongside, a product.")
    public List<ProductResponseDTO> getFrequentlyBoughtTogether(
            @ToolParam(description = "UUID of the reference product") UUID productId,
            @ToolParam(description = "max number of products to return") int limit) {
        try {
            return publish(recommendationService.frequentlyBoughtTogether(productId, limit));
        } catch (Exception ex) {
            log.warn("getFrequentlyBoughtTogether tool failed: {}", ex.getMessage());
            return List.of();
        }
    }

    @Tool(description = "Get the review sentiment summary (positive/neutral/negative counts and average rating) "
            + "for a specific product, given its product id.")
    public SentimentResponseDTO getProductSentiment(
            @ToolParam(description = "UUID of the product") UUID productId) {
        try {
            return sentimentService.analyzeProduct(productId, authHeader);
        } catch (Exception ex) {
            log.warn("getProductSentiment tool failed: {}", ex.getMessage());
            return null;
        }
    }

    @Tool(description = "Get an AI-written natural-language summary (what reviewers like/dislike) for a "
            + "specific product, given its product id. Prefer this over getProductReviews when the shopper "
            + "asks a general question like 'what do people think of this?' or 'are the reviews good?'.")
    public ReviewSummaryResponseDTO getProductReviewSummary(
            @ToolParam(description = "UUID of the product") UUID productId) {
        try {
            return reviewSummaryService.summarize(productId, authHeader);
        } catch (Exception ex) {
            log.warn("getProductReviewSummary tool failed: {}", ex.getMessage());
            return null;
        }
    }

    @Tool(description = "Answer a specific shopper question about ONE product (features, material, size, price, "
            + "what reviewers say about it), given its product id and the question. The answer is grounded only in "
            + "that product's own details and reviews; if answerable is false the information isn't available, so "
            + "say you don't know rather than guessing.")
    public ProductQaResponseDTO askProductQuestion(
            @ToolParam(description = "UUID of the product") UUID productId,
            @ToolParam(description = "the shopper's question about that product") String question) {
        try {
            return productQaService.ask(productId, question, null, authHeader);
        } catch (ResponseStatusException ex) {
            return new ProductQaResponseDTO(productId, ex.getReason() != null ? ex.getReason() : "Product not found",
                    false, List.of(), false);
        } catch (Exception ex) {
            log.warn("askProductQuestion tool failed: {}", ex.getMessage());
            return null;
        }
    }

    @Tool(description = "Get real customer reviews (text + rating) for a specific product, given its product id. "
            + "Use this when the shopper wants to read what others said, not just a sentiment summary.")
    public List<ReviewResponseDTO> getProductReviews(
            @ToolParam(description = "UUID of the product") UUID productId,
            @ToolParam(description = "max number of reviews to return") int limit) {
        try {
            log.info("Calling {} service -> getProductReviews(productId={})", "msvc-reviews", productId);
            List<ReviewResponseDTO> reviews = reviewCache.getReviews(productId, authHeader);
            List<ReviewResponseDTO> result = reviews.size() > limit ? reviews.subList(0, limit) : reviews;
            log.info("getProductReviews tool returning {} review(s)", result.size());
            return result;
        } catch (Exception ex) {
            log.warn("getProductReviews tool failed: {}", ex.getMessage());
            return List.of();
        }
    }

    @Tool(description = "Get the CURRENT shopper's own recent orders (order id, date, total, status). Use this "
            + "when they ask about 'my orders' or 'my order history'.")
    public List<OrderResponseDTO> getMyOrders(
            @ToolParam(description = "max number of orders to return") int limit) {
        if (currentUserId == null) {
            return List.of();
        }
        try {
            log.info("Calling {} service -> getMyOrders(userId={}, limit={})", "msvc-orders", currentUserId, limit);
            List<OrderResponseDTO> orders = orderClient.getOrdersByUser(currentUserId, authHeader);
            List<OrderResponseDTO> result = orders.stream()
                    .sorted(Comparator.comparing(OrderResponseDTO::getOrderDate,
                            Comparator.nullsLast(Comparator.reverseOrder())))
                    .limit(Math.max(limit, 1))
                    .toList();
            log.info("getMyOrders tool returning {} order(s)", result.size());
            return result;
        } catch (Exception ex) {
            log.warn("getMyOrders tool failed: {}", ex.getMessage());
            return List.of();
        }
    }

    @Tool(description = "Get the shipment tracking timeline (status, carrier, tracking number, per-seller-item "
            + "progress) for one of the CURRENT shopper's own orders, given its order id.")
    public List<SellerOrderTrackingDTO> trackOrder(
            @ToolParam(description = "UUID of the order to track") UUID orderId) {
        try {
            log.info("Calling {} service -> trackOrder(orderId={})", "msvc-seller", orderId);
            List<SellerOrderTrackingDTO> result = sellerOrderClient.track(orderId, authHeader);
            log.info("trackOrder tool returning {} tracking entry(ies)", result.size());
            return result;
        } catch (Exception ex) {
            log.warn("trackOrder tool failed: {}", ex.getMessage());
            return List.of();
        }
    }

    @Tool(description = "Cancel one of the CURRENT shopper's own orders, given its order id. Only works while "
            + "the order hasn't shipped yet. Returns a short confirmation or failure message.")
    public String cancelOrder(@ToolParam(description = "UUID of the order to cancel") UUID orderId) {
        try {
            log.info("Calling {} service -> cancelOrder(orderId={})", "msvc-orders", orderId);
            orderClient.cancelOrder(orderId, authHeader);
            log.info("cancelOrder tool succeeded for orderId={}", orderId);
            return "Your order has been cancelled and any online payment will be refunded automatically.";
        } catch (ResponseStatusException ex) {
            return ex.getReason() != null ? ex.getReason() : "That order can't be cancelled right now.";
        } catch (Exception ex) {
            log.warn("cancelOrder tool failed: {}", ex.getMessage());
            return "Sorry, I couldn't cancel that order right now. Please try again later.";
        }
    }

    @Tool(description = "Request a return for a delivered item, given its seller-order-line id (from trackOrder) "
            + "and a reason. Only works once that item has been DELIVERED. Returns a short confirmation or "
            + "failure message.")
    public String requestOrderReturn(
            @ToolParam(description = "UUID of the seller order line to return (from trackOrder)") UUID sellerOrderLineId,
            @ToolParam(description = "reason for the return") String reason) {
        try {
            log.info("Calling {} service -> requestOrderReturn(sellerOrderLineId={})", "msvc-seller", sellerOrderLineId);
            sellerOrderClient.requestReturn(sellerOrderLineId, new ReasonRequestDTO(reason), authHeader);
            log.info("requestOrderReturn tool succeeded for sellerOrderLineId={}", sellerOrderLineId);
            return "Your return request has been submitted and is waiting for the seller's approval.";
        } catch (ResponseStatusException ex) {
            return ex.getReason() != null ? ex.getReason() : "That item can't be returned right now.";
        } catch (Exception ex) {
            log.warn("requestOrderReturn tool failed: {}", ex.getMessage());
            return "Sorry, I couldn't submit that return request right now. Please try again later.";
        }
    }

    @Tool(description = "Get the refund status/history for one of the CURRENT shopper's own orders, given its "
            + "order id. Use this when they ask 'where is my refund'.")
    public List<RefundResponseDTO> getRefundStatus(
            @ToolParam(description = "UUID of the order") UUID orderId) {
        try {
            log.info("Calling {} service -> getRefundStatus(orderId={})", "msvc-refunds", orderId);
            List<RefundResponseDTO> result = refundClient.findByOrderId(orderId, authHeader);
            log.info("getRefundStatus tool returning {} refund record(s)", result.size());
            return result;
        } catch (Exception ex) {
            log.warn("getRefundStatus tool failed: {}", ex.getMessage());
            return List.of();
        }
    }

    @Tool(description = "DRAFT (does not submit) a return/refund request for one of the CURRENT shopper's own "
            + "orders: checks which items are eligible (DELIVERED and not already returned), proposes a "
            + "cleaned-up reason and lists the exact steps. Use this FIRST when the shopper wants to return "
            + "something; show them the draft, and only call requestOrderReturn after they confirm.")
    public Map<String, Object> draftReturnRequest(
            @ToolParam(description = "UUID of the order") UUID orderId,
            @ToolParam(description = "the shopper's reason in their own words") String reason) {
        Map<String, Object> draft = new LinkedHashMap<>();
        try {
            List<SellerOrderTrackingDTO> lines = sellerOrderClient.track(orderId, authHeader);
            List<Map<String, Object>> eligible = new java.util.ArrayList<>();
            List<String> blocked = new java.util.ArrayList<>();
            for (SellerOrderTrackingDTO l : lines) {
                if ("DELIVERED".equals(l.getStatus())) {
                    Map<String, Object> item = new LinkedHashMap<>();
                    item.put("sellerOrderLineId", l.getId());
                    item.put("productId", l.getProductId());
                    item.put("quantity", l.getQuantity());
                    item.put("lineTotal", l.getLineTotal());
                    eligible.add(item);
                } else {
                    blocked.add("Item " + l.getId() + " is " + l.getStatus() + " - "
                            + ("RETURN_REQUESTED".equals(l.getStatus()) || "RETURNED".equals(l.getStatus())
                                    ? "a return was already raised."
                                    : "returns open only after delivery (use cancel if it hasn't shipped)."));
                }
            }
            draft.put("orderId", orderId);
            draft.put("submitted", false);
            draft.put("eligibleItems", eligible);
            draft.put("notEligible", blocked);
            draft.put("proposedReason", reason == null || reason.isBlank() ? "Not specified" : reason.trim());
            draft.put("nextSteps", eligible.isEmpty()
                    ? List.of("Nothing in this order can be returned right now.",
                            "If something is wrong with the order, offer to escalate to human support.")
                    : List.of("Confirm which item(s) to return and the reason with the shopper.",
                            "Call requestOrderReturn with each sellerOrderLineId and the reason.",
                            "The seller reviews the request; the refund is processed after approval."));
            return draft;
        } catch (Exception ex) {
            log.warn("draftReturnRequest tool failed: {}", ex.getMessage());
            draft.put("submitted", false);
            draft.put("error", "Couldn't look up that order right now.");
            return draft;
        }
    }

    @Tool(description = "Escalate the conversation to a HUMAN support admin (sends them a notification with a "
            + "summary). Use when the shopper is stuck, angry, the return/refund can't be handled by you, "
            + "or they explicitly ask for a human. Returns a confirmation message.")
    public String escalateToHumanSupport(
            @ToolParam(description = "short summary of the problem and what was already tried") String summary,
            @ToolParam(description = "UUID of the related order, if any", required = false) UUID orderId) {
        String who = currentUserId != null ? currentUserId.toString() : "unknown user";
        String body = "Customer " + who + (orderId != null ? " (order " + orderId + ")" : "") + " needs help: "
                + (summary == null ? "no summary" : summary.trim());
        boolean ok = internalNotifier.toRole("ADMIN", "Support escalation from chatbot", body, "SYSTEM");
        return ok ? "I've passed this to our support team with the details - they'll get back to you soon."
                : "Sorry, I couldn't reach the support team right now. Please try again in a few minutes.";
    }

    @Tool(description = "Get the contents of the CURRENT shopper's own cart (items, quantities, prices, total).")
    public CartResponseDTO getMyCart() {
        try {
            log.info("Calling {} service -> getMyCart()", "msvc-carts");
            CartResponseDTO result = cartClient.getMyCart(authHeader);
            log.info("getMyCart tool returning cart with {} item(s)", result != null && result.getItems() != null ? result.getItems().size() : 0);
            return result;
        } catch (Exception ex) {
            log.warn("getMyCart tool failed: {}", ex.getMessage());
            return null;
        }
    }

    @Tool(description = "Add a product to the CURRENT shopper's own cart, given its product id and a quantity. "
            + "Returns a short confirmation or failure message.")
    public String addToCart(
            @ToolParam(description = "UUID of the product to add") UUID productId,
            @ToolParam(description = "quantity to add") int quantity) {
        try {
            log.info("Calling {} service -> addToCart(productId={}, quantity={})", "msvc-carts", productId, quantity);
            cartClient.addItem(new CartItemCreateRequestDTO(productId, Math.max(quantity, 1)), authHeader);
            log.info("addToCart tool succeeded for productId={}", productId);
            return "Added to your cart!";
        } catch (ResponseStatusException ex) {
            return ex.getReason() != null ? ex.getReason() : "Could not add that to the cart right now.";
        } catch (Exception ex) {
            log.warn("addToCart tool failed: {}", ex.getMessage());
            return "Sorry, I couldn't add that to the cart right now. Please try again later.";
        }
    }

    @Tool(description = "Get the CURRENT shopper's own wishlist (saved products).")
    public WishlistResponseDTO getMyWishlist() {
        try {
            log.info("Calling {} service -> getMyWishlist()", "msvc-wishlist");
            WishlistResponseDTO result = wishlistClient.getMyWishlist(authHeader);
            log.info("getMyWishlist tool returning wishlist with {} item(s)", result != null && result.getItems() != null ? result.getItems().size() : 0);
            return result;
        } catch (Exception ex) {
            log.warn("getMyWishlist tool failed: {}", ex.getMessage());
            return null;
        }
    }

    @Tool(description = "Add a product to the CURRENT shopper's own wishlist, given its product id. Returns a "
            + "short confirmation or failure message.")
    public String addToWishlist(@ToolParam(description = "UUID of the product to save") UUID productId) {
        try {
            log.info("Calling {} service -> addToWishlist(productId={})", "msvc-wishlist", productId);
            wishlistClient.addItem(new WishlistItemCreateRequestDTO(productId), authHeader);
            log.info("addToWishlist tool succeeded for productId={}", productId);
            return "Saved to your wishlist!";
        } catch (ResponseStatusException ex) {
            return ex.getReason() != null ? ex.getReason() : "Could not add that to the wishlist right now.";
        } catch (Exception ex) {
            log.warn("addToWishlist tool failed: {}", ex.getMessage());
            return "Sorry, I couldn't add that to the wishlist right now. Please try again later.";
        }
    }

    @Tool(description = "Get the CURRENT shopper's own recent notifications (order updates, offers, etc).")
    public List<NotificationResponseDTO> getMyNotifications(
            @ToolParam(description = "max number of notifications to return") int limit) {
        try {
            log.info("Calling {} service -> getMyNotifications(limit={})", "msvc-notifications", limit);
            List<NotificationResponseDTO> notifications = notificationClient.getMy(authHeader);
            List<NotificationResponseDTO> result = notifications.size() > limit ? notifications.subList(0, limit) : notifications;
            log.info("getMyNotifications tool returning {} notification(s)", result.size());
            return result;
        } catch (Exception ex) {
            log.warn("getMyNotifications tool failed: {}", ex.getMessage());
            return List.of();
        }
    }

    @Tool(description = "Get the count of the CURRENT shopper's unread notifications.")
    public Long getUnreadNotificationCount() {
        try {
            log.info("Calling {} service -> getUnreadNotificationCount()", "msvc-notifications");
            Long count = notificationClient.getMyUnreadCount(authHeader);
            long result = count != null ? count : 0L;
            log.info("getUnreadNotificationCount tool returning count={}", result);
            return result;
        } catch (Exception ex) {
            log.warn("getUnreadNotificationCount tool failed: {}", ex.getMessage());
            return 0L;
        }
    }

    @Tool(description = "Get the CURRENT shopper's own basic account info (username, email). Use this when they "
            + "ask about 'my account' or 'my profile'.")
    public UserContactDTO getMyProfile() {
        if (currentUserId == null) {
            return null;
        }
        try {
            log.info("Calling {} service -> getMyProfile(userId={})", "msvc-users", currentUserId);
            UserContactDTO result = userClient.findById(currentUserId, authHeader);
            log.info("getMyProfile tool returning profile found={}", result != null);
            return result;
        } catch (Exception ex) {
            log.warn("getMyProfile tool failed: {}", ex.getMessage());
            return null;
        }
    }

    @Tool(description = "Get factual information about LoomNest's store policies — payment methods, order "
            + "cancellation, returns, refunds, or order tracking. Use this for general policy questions that "
            + "are not about one specific order.")
    public String getStorePolicyInfo(
            @ToolParam(description = "policy topic or the shopper's policy question, e.g. payment, cancellation, "
                    + "returns, refund, tracking, warranty, shipping charges")
            String topic) {
        if (policyKnowledge != null && !isCoreTopic(topic)) {
            try {
                RagDtos.PolicyAnswer rag = policyKnowledge.answerTrusted(topic);
                if (rag != null && rag.answer() != null && !rag.answer().isBlank()) {
                    return rag.answer();
                }
            } catch (Exception ex) {
                log.warn("Policy RAG lookup failed, using static policy text: {}", ex.getMessage());
            }
        }
        return storePolicyText(topic);
    }

    private static boolean isCoreTopic(String topic) {
        String key = topic == null ? "" : topic.trim().toLowerCase();
        return key.contains("pay") || key.contains("cancel") || key.contains("return") || key.contains("refund")
                || key.contains("track") || key.contains("shipping") || key.contains("delivery");
    }

    /** Static so the read-only MCP tool set can share the exact same policy text without a per-user instance. */
    public static String storePolicyText(String topic) {
        String key = topic == null ? "" : topic.trim().toLowerCase();
        if (key.contains("pay")) {
            return "We accept UPI, Credit/Debit Card, NetBanking, Wallets and Cash on Delivery (COD). Online "
                    + "payments are processed securely through Razorpay.";
        }
        if (key.contains("cancel")) {
            return "You can cancel an order yourself as long as it hasn't shipped yet (while it's still "
                    + "Pending or Confirmed). Once a seller ships an item it can no longer be cancelled through "
                    + "self-service — you'd request a return instead once it's delivered.";
        }
        if (key.contains("return")) {
            return "Returns can be requested once an item has actually been DELIVERED. The seller then reviews "
                    + "and approves each return request.";
        }
        if (key.contains("refund")) {
            return "For online payments, a refund is triggered automatically when an order is cancelled, a "
                    + "seller declines it, or a return is approved. Cash-on-delivery orders have nothing to "
                    + "refund unless a return is approved after delivery.";
        }
        if (key.contains("track") || key.contains("shipping") || key.contains("delivery")) {
            return "LoomNest is a multi-seller marketplace, so each item in an order ships independently from "
                    + "its own seller with its own status timeline (placed, accepted, shipped, delivered) and, "
                    + "once shipped, a carrier name and tracking number when available. Ask me to track a "
                    + "specific order and I'll pull up the latest status.";
        }
        return "LoomNest accepts UPI/Card/NetBanking/Wallet/COD payments, lets you cancel an order before it "
                + "ships, supports returns after delivery (subject to seller approval), and auto-refunds online "
                + "payments on cancellation or an approved return. Ask me about a specific topic (payment, "
                + "cancellation, returns, refunds, or tracking) for more detail.";
    }

    /** Products accumulated across every tool call made during this chat turn (for the UI's product cards). */
    public List<ProductResponseDTO> accumulatedProducts() {
        return List.copyOf(accumulated.values());
    }

    private List<ProductResponseDTO> publish(List<ProductResponseDTO> results) {
        for (ProductResponseDTO product : results) {
            if (accumulated.size() >= MAX_ACCUMULATED) {
                break;
            }
            accumulated.putIfAbsent(product.getId(), product);
        }
        onProductsResolved.accept(List.copyOf(accumulated.values()));
        return results;
    }
}
