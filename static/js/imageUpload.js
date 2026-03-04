/**
 * NOPE Chat — Image upload handling
 * File validation, base64 encoding, preview management.
 */

let pendingFile = null;

function validateFile(file) {
  if (!ALLOWED_MIME_TYPES.includes(file.type)) {
    return 'Unsupported file type. Please use JPEG, PNG, GIF, or WebP.';
  }
  if (file.size > MAX_FILE_SIZE) {
    return 'Image is too large (max 20 MB).';
  }
  return null;
}

function readFileAsBase64(file) {
  return new Promise(function(resolve, reject) {
    var reader = new FileReader();
    reader.onload = function() {
      // result is "data:<type>;base64,<data>" — extract the base64 part
      var base64 = reader.result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = function() {
      reject(new Error('Failed to read file'));
    };
    reader.readAsDataURL(file);
  });
}

function showImagePreview(file) {
  var preview = document.getElementById('image-preview');
  var thumbnail = document.getElementById('preview-thumbnail');
  var filename = document.getElementById('preview-filename');

  thumbnail.src = URL.createObjectURL(file);
  filename.textContent = file.name;
  preview.style.display = 'flex';
}

function clearImagePreview() {
  var preview = document.getElementById('image-preview');
  var thumbnail = document.getElementById('preview-thumbnail');

  if (thumbnail.src) {
    URL.revokeObjectURL(thumbnail.src);
  }
  thumbnail.src = '';
  preview.style.display = 'none';
  pendingFile = null;

  // Reset file input
  var fileInput = document.getElementById('file-input');
  if (fileInput) fileInput.value = '';
}

function handleFileSelect(file) {
  var error = validateFile(file);
  if (error) {
    alert(error);
    return;
  }
  pendingFile = file;
  showImagePreview(file);
}
