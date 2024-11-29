// script.js
function handleContinue() {
  // Redirect to the new page with no content in the middle
  window.location.href = "/next_page";  // This will open the new page
}

function handlePreview() {
  console.log("Preview button clicked"); // Debugging log
  window.location.href = "/preview_page"; // Redirect to preview page
}

async function handleSubmit() {
  const form = document.getElementById("resumeForm");
  const formData = new FormData(form);

  try {
    // Send the form data to the server using fetch
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });

    if (response.redirected) {
      // If the response contains a redirect, navigate to the new URL
      window.location.href = response.url;
    } else if (!response.ok) {
      const errorText = await response.text();
      alert(`Error: ${errorText}`);
    }
  } catch (error) {
    console.error("Error uploading file:", error);
    alert("An error occurred during submission. Please try again.");
  }
}

