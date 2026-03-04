/**
 * NOPE Chat — Runtime configuration
 * N8N_WEBHOOK_URL is injected by the /config.js server endpoint.
 */

const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20 MB
const ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];

function getWebhookUrl() {
  return (window.__NOPE_CONFIG__ && window.__NOPE_CONFIG__.N8N_WEBHOOK_URL) || '';
}

function getSessionId() {
  let id = localStorage.getItem('nope_session_id');
  if (!id) {
    id = 'session_' + Math.random().toString(36).substring(2) + Date.now().toString(36);
    localStorage.setItem('nope_session_id', id);
  }
  return id;
}
