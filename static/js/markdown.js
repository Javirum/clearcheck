/**
 * NOPE Chat — Lightweight markdown-to-HTML renderer
 * Supports: bold, italic, links, line breaks, unordered/ordered lists, horizontal rules.
 * All input is HTML-escaped first to prevent XSS.
 */

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function renderMarkdown(text) {
  // Escape HTML first
  let html = escapeHtml(text);

  // Horizontal rules
  html = html.replace(/^---+$/gm, '<hr>');

  // Bold: **text** or __text__
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');

  // Italic: *text* or _text_
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  html = html.replace(/(?<!\w)_(.+?)_(?!\w)/g, '<em>$1</em>');

  // Links: [text](url)
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');

  // Unordered lists: lines starting with - or *
  html = html.replace(/^(?:[-*])\s+(.+)$/gm, '<li>$1</li>');
  html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, '<ul>$1</ul>');

  // Ordered lists: lines starting with 1. 2. etc.
  html = html.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');
  // Wrap consecutive <li> not already in <ul> into <ol>
  html = html.replace(/(?<!<\/ul>)((?:<li>.*<\/li>\n?)+)/g, function(match) {
    // Only wrap if not already inside a <ul>
    if (match.includes('<ul>')) return match;
    return '<ol>' + match + '</ol>';
  });

  // Line breaks: double newline → paragraph break, single newline → <br>
  html = html.replace(/\n\n+/g, '</p><p>');
  html = html.replace(/\n/g, '<br>');

  // Wrap in paragraph
  html = '<p>' + html + '</p>';

  // Clean up empty paragraphs
  html = html.replace(/<p>\s*<\/p>/g, '');

  return html;
}
