function startReview() {
    window.location.href = "/index1"; // Redirect to the index page
}

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

//     window.location.href = "/review1";
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
            window.location.href = "/review1";
        } else {
            alert("Error processing your request. ");
        }
    
    })
    .catch(error => {
        console.error("Error:", error);
        alert("Failed to analyze the code.");
    });
}

function goBack() {
    window.location.href = "/intro1";
}

function goBack() {
    window.location.href = "/index1";
}
