/**
 * Insight into RAG - Minimal & High-Performance Web UI
 * Tập trung 100% vào Hỏi đáp Trí tuệ và Lựa chọn Mô hình LLM.
 * Hoàn toàn loại bỏ bộ chọn Embedding Model khỏi giao diện người dùng.
 */

document.addEventListener("DOMContentLoaded", () => {
  // ============================================================
  // 1. STATE MANAGEMENT
  // ============================================================
  let isGenerating = false;
  let currentAbortController = null;
  // Lưu lịch sử hội thoại (tối đa 10 lượt)
  const MAX_HISTORY = 10;
  let conversationHistory = [];

  // DOM Elements
  const systemStatus = document.getElementById("system-status");

  const chatThread = document.getElementById("chat-thread");
  const welcomeHero = document.getElementById("welcome-hero");
  const chatForm = document.getElementById("chat-form");
  const queryInput = document.getElementById("query-input");
  const btnSubmit = document.getElementById("btn-submit");
  const btnAbort = document.getElementById("btn-abort");
  const btnSpinner = document.getElementById("btn-spinner");
  const sendIcon = document.getElementById("send-icon");



  // ============================================================
  // 2. KHỞI TẠO & ĐỒNG BỘ CẤU HÌNH SERVER
  // ============================================================
  async function initApp() {
    try {
      const res = await fetch("/api/config");
      if (res.ok) {
        const data = await res.json();
        systemStatus.innerHTML = `
          <span class="status-dot"></span>
          <span class="status-text">Sẵn sàng (${data.doc_count || 0} tài liệu)</span>
        `;
      } else {
        throw new Error("Lỗi nạp config");
      }
    } catch (e) {
      systemStatus.innerHTML = `
        <span class="status-dot" style="background:#F59E0B;box-shadow:0 0 8px #F59E0B;"></span>
        <span class="status-text" style="color:#F59E0B;">Ngoại tuyến (Offline)</span>
      `;
    }
  }

  // ============================================================
  // 4. XỬ LÝ GỬI CÂU HỎI VÀ HIỂN THỊ CHAT
  // ============================================================
  // Auto-resize textarea
  queryInput.addEventListener("input", () => {
    queryInput.style.height = "auto";
    queryInput.style.height = Math.min(queryInput.scrollHeight, 160) + "px";
  });

  // Phím tắt Enter để gửi, Shift+Enter xuống dòng
  queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isGenerating && queryInput.value.trim()) {
        chatForm.dispatchEvent(new Event("submit"));
      }
    }
  });

  // Quick Prompt Chips
  document.addEventListener("click", (e) => {
    const chip = e.target.closest(".prompt-chip");
    if (chip) {
      const query = chip.getAttribute("data-query");
      if (query && !isGenerating) {
        queryInput.value = query;
        chatForm.dispatchEvent(new Event("submit"));
      }
    }
  });



  // Simple Markdown Parser
  function parseMarkdown(text) {
    if (!text) return "";
    let safe = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // Code blocks
    safe = safe.replace(/```([a-zA-Z0-9]*)\n([\s\S]*?)```/g, (match, lang, code) => {
      return `<pre><code>${code.trim()}</code></pre>`;
    });

    // Inline code
    safe = safe.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Bold & Italic
    safe = safe.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    safe = safe.replace(/\*([^*]+)\*/g, "<em>$1</em>");

    // Paragraphs & lines
    const paragraphs = safe.split(/\n\n+/);
    return paragraphs
      .map(p => {
        p = p.trim();
        if (p.startsWith("<pre>")) return p;
        if (p.startsWith("- ") || p.startsWith("* ")) {
          const items = p.split(/\n[-*]\s+/).filter(Boolean);
          return "<ul>" + items.map(it => `<li>${it.replace(/^[-*]\s+/, "")}</li>`).join("") + "</ul>";
        }
        if (/^\d+\.\s+/.test(p)) {
          const items = p.split(/\n\d+\.\s+/).filter(Boolean);
          return "<ol>" + items.map(it => `<li>${it.replace(/^\d+\.\s+/, "")}</li>`).join("") + "</ol>";
        }
        return `<p>${p.replace(/\n/g, "<br>")}</p>`;
      })
      .join("");
  }

  // Thêm tin nhắn vào khung chat
  function appendMessage(sender, text, metaInfo = null, contexts = []) {
    if (welcomeHero && welcomeHero.parentNode === chatThread) {
      welcomeHero.style.display = "none";
    }

    const messageEl = document.createElement("div");
    messageEl.className = `chat-message ${sender}-message`;

    const avatarEl = document.createElement("div");
    avatarEl.className = "message-avatar";
    avatarEl.textContent = sender === "user" ? "U" : "AI";

    const contentWrapper = document.createElement("div");
    contentWrapper.className = "message-content-wrapper";

    const bubbleEl = document.createElement("div");
    bubbleEl.className = "message-bubble";

    if (sender === "user") {
      bubbleEl.textContent = text;
    } else {
      bubbleEl.innerHTML = parseMarkdown(text);

      // Thêm Citations nếu có
      if (contexts && contexts.length > 0) {
        const citationsBox = document.createElement("div");
        citationsBox.className = "citations-box";

        const toggleBtn = document.createElement("button");
        toggleBtn.type = "button";
        toggleBtn.className = "citations-toggle";
        toggleBtn.innerHTML = `
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
            <polyline points="14 2 14 8 20 8"></polyline>
          </svg>
          <span>Xem ${contexts.length} nguồn tham khảo & trích dẫn</span>
        `;

        const citationsList = document.createElement("div");
        citationsList.className = "citations-list";

        contexts.forEach((c) => {
          const card = document.createElement("div");
          card.className = "citation-card";

          // Xử lý trang hoặc điều khoản
          let locationText = "";
          if (c.page) {
            locationText = `Trang ${c.page}`;
          } else if (c.section) {
            locationText = c.section;
          } else {
            locationText = "Văn bản";
          }

          card.innerHTML = `
            <div class="citation-header">
              <span class="citation-source">📄 ${c.source}</span>
              <span class="citation-page">${locationText}</span>
            </div>
            <div class="citation-snippet">${c.text}</div>
          `;
          citationsList.appendChild(card);
        });

        toggleBtn.addEventListener("click", () => {
          const isExp = citationsList.classList.toggle("expanded");
          toggleBtn.querySelector("span").textContent = isExp
            ? `Ẩn ${contexts.length} nguồn tham khảo`
            : `Xem ${contexts.length} nguồn tham khảo & trích dẫn`;
        });

        citationsBox.appendChild(toggleBtn);
        citationsBox.appendChild(citationsList);
        bubbleEl.appendChild(citationsBox);
      }
    }

    contentWrapper.appendChild(bubbleEl);

    // Meta footer
    if (metaInfo) {
      const metaEl = document.createElement("div");
      metaEl.className = "message-meta";
      metaEl.textContent = metaInfo;
      contentWrapper.appendChild(metaEl);
    }

    messageEl.appendChild(avatarEl);
    messageEl.appendChild(contentWrapper);
    chatThread.appendChild(messageEl);

    // Scroll to bottom
    chatThread.scrollTop = chatThread.scrollHeight;
    return messageEl;
  }

  // Hiển thị bóng chat Loading
  function showLoadingIndicator() {
    if (welcomeHero && welcomeHero.parentNode === chatThread) {
      welcomeHero.style.display = "none";
    }

    const loadingEl = document.createElement("div");
    loadingEl.className = "chat-message ai-message";
    loadingEl.id = "loading-message";

    loadingEl.innerHTML = `
      <div class="message-avatar">AI</div>
      <div class="message-content-wrapper">
        <div class="message-bubble typing-dots">
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span style="font-size:0.82rem;color:var(--text-muted);margin-left:8px;">AI đang tra cứu tài liệu và suy luận...</span>
        </div>
      </div>
    `;

    chatThread.appendChild(loadingEl);
    chatThread.scrollTop = chatThread.scrollHeight;
    return loadingEl;
  }

  function removeLoadingIndicator() {
    const loadingEl = document.getElementById("loading-message");
    if (loadingEl) loadingEl.remove();
  }

  // ============================================================
  // 5. GỬI YÊU CẦU TRA CỨU QUA API /api/chat
  // ============================================================
  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query || isGenerating) return;

    // Reset input
    queryInput.value = "";
    queryInput.style.height = "auto";

    // Append User message
    appendMessage("user", query);

    // Trạng thái đang sinh
    isGenerating = true;
    btnSubmit.disabled = true;
    btnSpinner.style.display = "inline-block";
    sendIcon.style.display = "none";
    btnAbort.style.display = "flex";

    const loadingEl = showLoadingIndicator();
    currentAbortController = new AbortController();

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: currentAbortController.signal,
        body: JSON.stringify({
          query: query,
          conversation_history: conversationHistory.slice(-MAX_HISTORY)
        }),
      });

      removeLoadingIndicator();

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errMsg = errorData.detail || `Lỗi máy chủ (${response.status})`;
        appendMessage(
          "ai",
          `⚠️ **Không thể hoàn tất tra cứu:** ${errMsg}\n\n*Gợi ý:* Vui lòng kiểm tra lại API Key trong mục **API Key** ở góc trên màn hình.`
        );
        return;
      }

      const data = await response.json();
      const meta = `${data.model || "Default Model"} • ${data.latency_ms || 0} ms • Hybrid RRF`;
      appendMessage("ai", data.answer || "Không có phản hồi từ mô hình.", meta, data.contexts || []);

      // Cập nhật lịch sử hội thoại
      conversationHistory.push({ role: "user", content: query });
      conversationHistory.push({ role: "assistant", content: data.answer || "" });
      // Giới hạn độ dài lịch sử
      if (conversationHistory.length > MAX_HISTORY * 2) {
        conversationHistory = conversationHistory.slice(-MAX_HISTORY * 2);
      }

    } catch (err) {
      removeLoadingIndicator();
      if (err.name === "AbortError") {
        appendMessage("ai", "⏹️ *Đã dừng tạo câu trả lời.*");
      } else {
        appendMessage("ai", `⚠️ **Đã xảy ra sự cố kết nối:** ${err.message}`);
      }
    } finally {
      isGenerating = false;
      currentAbortController = null;
      btnSubmit.disabled = false;
      btnSpinner.style.display = "none";
      sendIcon.style.display = "block";
      btnAbort.style.display = "none";
      queryInput.focus();
    }
  });

  btnAbort.addEventListener("click", () => {
    if (isGenerating && currentAbortController) {
      currentAbortController.abort();
    }
  });

  // Khởi động ứng dụng
  initApp();
});
