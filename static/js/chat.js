/**
 * NOPE Chat — Core chat logic
 * Sends messages to n8n Chat Trigger webhook, renders responses.
 */

var HISTORY_KEY = 'nope_chat_history';

function getChatHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
  } catch (_) {
    return [];
  }
}

function saveChatHistory(messages) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(messages));
  } catch (_) {
    // Storage full — silently ignore
  }
}

function appendMessage(role, html, imageData) {
  var container = document.getElementById('messages');
  var wrapper = document.createElement('div');
  wrapper.className = 'message ' + role;

  var bubble = document.createElement('div');
  bubble.className = 'bubble';

  if (imageData) {
    var img = document.createElement('img');
    img.src = 'data:' + imageData.type + ';base64,' + imageData.data;
    img.className = 'message-image';
    img.alt = imageData.name || 'Uploaded image';
    bubble.appendChild(img);
  }

  var content = document.createElement('div');
  content.className = 'message-content';
  content.innerHTML = html;
  bubble.appendChild(content);

  wrapper.appendChild(bubble);
  container.appendChild(wrapper);
  scrollToBottom();
}

function showTyping() {
  var container = document.getElementById('messages');
  var wrapper = document.createElement('div');
  wrapper.className = 'message bot';
  wrapper.id = 'typing-indicator';

  var bubble = document.createElement('div');
  bubble.className = 'bubble typing';
  bubble.innerHTML = '<span class="dot"></span><span class="dot"></span><span class="dot"></span>';

  wrapper.appendChild(bubble);
  container.appendChild(wrapper);
  scrollToBottom();
}

function hideTyping() {
  var el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

function scrollToBottom() {
  var container = document.getElementById('messages');
  container.scrollTop = container.scrollHeight;
}

function autoResize(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
}

async function sendMessage() {
  var input = document.getElementById('chat-input');
  var text = input.value.trim();

  if (!text && !pendingFile) return;

  var webhookUrl = getWebhookUrl();
  if (!webhookUrl) {
    appendMessage('bot', renderMarkdown('**Configuration error:** The chat service is not configured yet. Please set the N8N_WEBHOOK_URL environment variable.'));
    return;
  }

  // Build display text
  var displayText = text || '(image)';
  var filePayload = null;
  var imagePreviewData = null;

  // Handle pending image
  if (pendingFile) {
    try {
      var base64 = await readFileAsBase64(pendingFile);
      filePayload = {
        name: pendingFile.name,
        type: pendingFile.type,
        size: pendingFile.size,
        data: base64
      };
      imagePreviewData = { name: pendingFile.name, type: pendingFile.type, data: base64 };
    } catch (_) {
      appendMessage('bot', renderMarkdown('**Error:** Could not read the image file. Please try again.'));
      return;
    }
    clearImagePreview();
  }

  // Show user message
  appendMessage('user', escapeHtml(displayText), imagePreviewData);

  // Save to history
  var history = getChatHistory();
  history.push({ role: 'user', text: displayText, image: imagePreviewData ? { name: imagePreviewData.name, type: imagePreviewData.type } : null });
  saveChatHistory(history);

  // Clear input
  input.value = '';
  autoResize(input);

  // Build request
  var body = {
    action: 'sendMessage',
    chatInput: text || 'Please analyze this image.',
    sessionId: getSessionId()
  };
  if (filePayload) {
    body.files = [filePayload];
  }

  // Show typing
  showTyping();
  setSendEnabled(false);

  try {
    var resp = await fetch(webhookUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    hideTyping();

    if (!resp.ok) {
      throw new Error('Server returned ' + resp.status);
    }

    var data = await resp.json();
    var output = data.output || 'No response received.';
    var html = renderMarkdown(output);

    appendMessage('bot', html);

    // Save bot response to history
    history = getChatHistory();
    history.push({ role: 'bot', text: output });
    saveChatHistory(history);

  } catch (err) {
    hideTyping();
    appendMessage('bot', renderMarkdown('**Oops!** Something went wrong. Please try again in a moment.\n\n_' + escapeHtml(err.message) + '_'));
  }

  setSendEnabled(true);
}

function setSendEnabled(enabled) {
  var btn = document.getElementById('send-btn');
  if (btn) btn.disabled = !enabled;
}

function loadHistory() {
  var history = getChatHistory();
  for (var i = 0; i < history.length; i++) {
    var msg = history[i];
    if (msg.role === 'user') {
      appendMessage('user', escapeHtml(msg.text));
    } else {
      appendMessage('bot', renderMarkdown(msg.text));
    }
  }
}

function clearHistory() {
  localStorage.removeItem(HISTORY_KEY);
  document.getElementById('messages').innerHTML = '';
  showWelcome();
}

function showWelcome() {
  appendMessage('bot', renderMarkdown(
    "**Hey there! I'm NOPE.** &#128075;\n\n" +
    "Send me a claim, news headline, or forward you're suspicious about, and I'll check it for you.\n\n" +
    "You can also attach an image (screenshot, photo) using the &#128206; button.\n\n" +
    "_No judgment, just facts._"
  ));
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
  var input = document.getElementById('chat-input');
  var sendBtn = document.getElementById('send-btn');
  var fileInput = document.getElementById('file-input');
  var attachBtn = document.getElementById('attach-btn');
  var removeBtn = document.getElementById('remove-image');

  // Load history or show welcome
  var history = getChatHistory();
  if (history.length > 0) {
    loadHistory();
  } else {
    showWelcome();
  }

  // Send on button click
  sendBtn.addEventListener('click', sendMessage);

  // Send on Enter, newline on Shift+Enter
  input.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Auto-resize textarea
  input.addEventListener('input', function() {
    autoResize(input);
  });

  // File input
  attachBtn.addEventListener('click', function() {
    fileInput.click();
  });

  fileInput.addEventListener('change', function() {
    if (fileInput.files && fileInput.files[0]) {
      handleFileSelect(fileInput.files[0]);
    }
  });

  // Remove image preview
  removeBtn.addEventListener('click', clearImagePreview);
});
