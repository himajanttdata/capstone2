function startReview() {
    window.location.href = "/index2"; // Redirect to the index page
}

const toggleBtn = document.getElementById('toggle-btn');
const sidebar = document.getElementById('sidebar');
const body = document.body;

// Hamburger toggle
toggleBtn.addEventListener('click', () => {
sidebar.classList.toggle('active');
});
// Dark Mode toggle
function toggleDarkMode() {
body.classList.toggle('dark-mode');
// Optional: Save preference
localStorage.setItem('darkMode', body.classList.contains('dark-mode'));
}

// Load saved mode
window.onload = function () {
const savedMode = localStorage.getItem('darkMode');
if (savedMode === 'true') {
body.classList.add('dark-mode');
}
};

function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    if (sidebar.style.left === "0px") {
        sidebar.style.left = "-250px";
    } else {
        sidebar.style.left = "0px";
    }
}


document.addEventListener("DOMContentLoaded", () => {
    // Load saved theme
    if (localStorage.getItem("theme") === "dark") {
        document.body.classList.add("dark-mode");
    }

    // Attach dark mode toggle button
    const darkModeBtn = document.getElementById("darkModeToggle");
    if (darkModeBtn) {
        darkModeBtn.addEventListener("click", toggleDarkMode);
    }
});

// function analyzeCode() {
//     let githubURL = document.getElementById("github-url").value;
//     let files = document.getElementById("file-upload").files;

//     if (!githubURL && files.length === 0) {
//         alert("Please provide a GitHub URL or upload files.");
//         return;
//     }

//     // Store data and navigate to review page
//     sessionStorage.setItem("githubURL", githubURL);
//     sessionStorage.setItem("uploadedFiles", files.length);

//     window.location.href = "/review2";
// }
function analyzeCode() {
    let githubURL = document.getElementById("github-url").value;
    let files = document.getElementById("file-upload").files;

    if (!githubURL && files.length === 0) {
        alert("Please provide a GitHub URL or upload files.");
        return;
    }

    let formData = new FormData();
    if (githubURL) {
        formData.append("repo_url", githubURL);
    }
    
    for (let i = 0; i < files.length; i++) {
        formData.append("file", files[i]);
    }

    // Send data to backend
    fetch("/submit", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success){
            window.location.href = "/review2";
        } else {
            alert("Error processing your request. ");
        }
    
    })
    .catch(error => {
        console.error("Error:", error);
        alert("Failed to analyze the code.");
    });
}

async function fetchRepoFiles() {
    const githubURL = document.getElementById("github-url").value.trim();

    if (!githubURL) {
        alert("Please enter a GitHub repository URL.");
        return;
    }

    try {
        // ✅ Strip .git if present
        const cleanURL = githubURL.replace(/\.git$/, "");
        
        // ✅ Extract owner and repo from the URL
        const match = cleanURL.match(/^https:\/\/github\.com\/([^\/]+)\/([^\/]+)$/);
        if (!match) throw new Error("Invalid GitHub repository URL.");

        const owner = match[1];
        const repo = match[2];

        // ✅ GitHub API endpoint to get top-level contents
        const apiURL = `https://api.github.com/repos/${owner}/${repo}/contents`;

        const response = await fetch(apiURL);
        if (!response.ok) throw new Error("Failed to fetch repo contents");

        const files = await response.json();
        displayFiles(files);
    } catch (error) {
        console.error("Error:", error.message);
        alert("Failed to fetch files. Make sure the URL is public and correct.");
    }
}

function displayFiles(files) {
    const fileListDiv = document.getElementById("file-list");
    fileListDiv.innerHTML = "<h3>📁 Files in Repository:</h3>";

    const ul = document.createElement("ul");

    files.forEach(file => {
        const li = document.createElement("li");
        const link = document.createElement("a");
        link.href = file.html_url;
        link.textContent = file.name;
        link.target = "_blank";
        li.appendChild(link);
        ul.appendChild(li);
    });

    fileListDiv.appendChild(ul);
}



function goBack() {
    window.location.href = "/intro2";
}

function goBack() {
    window.location.href = "/index2";
}

// function downloadReviewReport() {
//     // Collect review content from the DOM
//     const bugs = document.getElementById("bug-list").innerText;
//     const optimizations = document.getElementById("optimization-list").innerText;
//     const docs = document.getElementById("documentation").innerText;
//     const reframing = document.getElementById("reframing-list").innerText;

//     // Create text content for the file
//     const content = `AI Code Review Report\n\n` +
//         `🪲 Detected Issues & Bugs:\n${bugs}\n\n` +
//         `🚀 Optimization Suggestions:\n${optimizations}\n\n` +
//         `📄 Auto-Generated Documentation:\n${docs}\n\n` +
//         `🔧 Code Reframing Suggestions:\n${reframing}`;

//     // Create a blob and download link
//     const blob = new Blob([content], { type: "text/plain" });
//     const url = URL.createObjectURL(blob);
//     const a = document.createElement("a");
//     a.href = url;
//     a.download = "AI_Code_Review_Report.txt";
//     document.body.appendChild(a);
//     a.click();
//     document.body.removeChild(a);
//     URL.revokeObjectURL(url);
// }
function downloadDOCX() {
    const reviewText = document.getElementById("review-text").innerText;
    fetch('/download_docx', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review: reviewText })
    })
    .then(response => response.blob())
    .then(blob => {
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = "Code_Review_Report.docx";
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    })
    .catch(error => console.error('Error downloading DOCX:', error));
}


