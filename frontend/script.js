
const CONFIG = {
    
    getApiUrl: () => {
        // For local development, point to the local server.
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            return 'http://127.0.0.1:8000';
        }
        
        
        return import.meta.env.VITE_API_URL || 'API_URL_NOT_SET';
    },
    
    
    isDevelopment: () => {
        return window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    },
    endpoints: {
        analyze: '/analyze',
        health: '/' // Health check endpoint
    }
};

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 DOM loaded, initializing application...');
    
    // File input handling
    const fileInput = document.getElementById('fileInput');
    const page1 = document.getElementById('page1');
    const page2 = document.getElementById('page2');
    const loading = document.getElementById('loading');
    const xrayImage = document.getElementById('xrayImage');
    const xrayPlaceholder = document.getElementById('xrayPlaceholder');
    const analysisResults = document.getElementById('analysisResults');
    const downloadBtn = document.getElementById('downloadBtn');

    // Debug: Check if all elements are found
    console.log('🔍 DOM Elements Check:', {
        fileInput: !!fileInput,
        page1: !!page1,
        page2: !!page2,
        loading: !!loading,
        xrayImage: !!xrayImage,
        xrayPlaceholder: !!xrayPlaceholder,
        analysisResults: !!analysisResults,
        downloadBtn: !!downloadBtn
    });

    // Check if fileInput exists before adding event listener
    if (!fileInput) {
        console.error('❌ File input element not found!');
        return;
    }

    // Show current configuration in console for debugging
    console.log('🔧 App Configuration:', {
        environment: CONFIG.isDevelopment() ? 'Development' : 'Production',
        apiUrl: CONFIG.getApiUrl(),
        hostname: window.location.hostname
    });// Handle file selection
fileInput.addEventListener('change', function(event) {
    console.log('🔍 File input changed, event triggered');
    const file = event.target.files[0];
    
    if (file) {
        console.log('📁 File selected:', file.name, 'Type:', file.type, 'Size:', file.size);
        
        // Validate file type
        const validTypes = ['image/png', 'image/jpg', 'image/jpeg'];
        if (!validTypes.includes(file.type)) {
            console.log('❌ Invalid file type:', file.type);
            alert('Please upload a PNG or JPG file.');
            return;
        }

        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            console.log('❌ File too large:', file.size);
            alert('File size should be less than 10MB.');
            return;
        }

        console.log('✅ File validation passed, showing loading screen');
        // Show loading screen
        loading.style.display = 'flex';
        
        // Create file reader to display image
        const reader = new FileReader();
        reader.onload = function(e) {
            console.log('📷 Image loaded by FileReader, transitioning to page 2');
            // Hide loading after a realistic delay
            setTimeout(() => {
                console.log('🔄 Starting page transition...');
                // Set the uploaded image
                xrayImage.src = e.target.result;
                xrayImage.style.display = 'block';
                xrayPlaceholder.style.display = 'none';
                
                // Hide page 1, show page 2
                page1.style.display = 'none';
                page2.style.display = 'block';
                loading.style.display = 'none';
                console.log('✅ Page transition completed, starting analysis...');
                
                // Call real API for analysis
                performAnalysis(file);
                
            }, 1500); // Realistic loading time
        };
        
        reader.onerror = function(error) {
            console.error('❌ FileReader error:', error);
            loading.style.display = 'none';
            alert('Error reading the file. Please try again.');
        };
        
        console.log('📖 Starting to read file...');
        reader.readAsDataURL(file);
    } else {
        console.log('❌ No file selected');
    }
    });

    // Real API analysis call
    async function performAnalysis(file) {
        console.log('🔬 Starting performAnalysis function...');
        // Show loading state in analysis box
        analysisResults.innerHTML = '<div style="text-align: center; color: #666;">Analyzing X-Ray scan...</div>';
        
        try {
            // Create FormData to send file to backend
            const formData = new FormData();
            formData.append('image', file);
            
            // Get API URL from configuration
            const apiUrl = CONFIG.getApiUrl();
            const endpoint = `${apiUrl}${CONFIG.endpoints.analyze}`;
            
            console.log('📡 Making API request to:', endpoint);
            
            // Call the backend API
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData,
                // Add headers for CORS if needed
                headers: {
                    // Don't set Content-Type - let browser set it with boundary for FormData
                }
            });
            
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`API request failed with status ${response.status}: ${errorText}`);
            }
            
            const data = await response.json();
            
            console.log('✅ Analysis completed successfully');
            
            // Store analysis data globally for PDF generation
            window.currentAnalysisData = data;
            
            // Display the results
            displayAnalysisResults(data);
            
            // Enable download button
            downloadBtn.disabled = false;
            
        } catch (error) {
            console.error('❌ Error analyzing image:', error);
            
            const apiUrl = CONFIG.getApiUrl();
            const isLocal = CONFIG.isDevelopment();
            
            let errorMessage = '';
            if (apiUrl === 'API_URL_NOT_SET') {
                errorMessage = 'API URL not configured. Please check deployment configuration.';
            } else {
                errorMessage = error.message;
            }
            
            analysisResults.innerHTML = `
                <div style="text-align: center; color: #d32f2f;">
                    <h3 style="margin-bottom: 15px;">Analysis Failed</h3>
                    <p>Error: ${errorMessage}</p>
                    <div style="font-size: 14px; margin-top: 10px; color: #666;">
                        <p><strong>Environment:</strong> ${isLocal ? 'Development' : 'Production'}</p>
                        ${isLocal ? 
                            '<p>Please ensure the backend server is running on port 8000.</p>' :
                            '<p>Please check if the backend service is running.</p>'
                        }
                    </div>
                </div>
            `;
        }
    }

    // Display analysis results
    function displayAnalysisResults(data) {
        console.log('🎨 Displaying analysis results:', data);
        const { probabilities, heatmap_image, report_text } = data;    // Create the results HTML
    let resultsHTML = '<div class="analysis-content">';
    
    // Top 5 Probabilities Section
    if (probabilities) {
        resultsHTML += '<h3>Top Observations:</h3>';
        resultsHTML += '<div class="probabilities-container">';
        
        Object.entries(probabilities).forEach(([condition, probability], index) => {
            const percentage = (probability * 100).toFixed(1);
            const barWidth = Math.max(percentage * 0.8, 5); // Minimum 5% width for visibility
            
            resultsHTML += `
                <div class="probability-item">
                    <div class="probability-header">
                        <span class="condition-name">${condition.replace(/_/g, ' ').toUpperCase()}</span>
                        <span class="confidence-score">${percentage}%</span>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-bar" style="width: ${barWidth}%"></div>
                    </div>
                </div>
            `;
        });
        resultsHTML += '</div>';
    }
    
    // Heatmap Section
    if (heatmap_image) {
        resultsHTML += '<h3>Heatmap Visualization:</h3>';
        resultsHTML += `
            <div class="heatmap-container">
                <img src="${heatmap_image}" class="heatmap-image" alt="Heatmap Visualization" />
                <p style="font-size: 12px; color: #666; margin-top: 8px; font-style: italic;">
                    Areas of highest model attention highlighted
                </p>
            </div>
        `;
    }
    
    // AI Report Section
    if (report_text) {
        resultsHTML += '<h3 style="margin-bottom: 15px; color: #4a4a8a;">AI Summary:</h3>';
        
        // Parse the report text for better formatting
        const formattedReport = formatAIReport(report_text);
        
        resultsHTML += `
            <div class="ai-summary-section">
                ${formattedReport}
            </div>
        `;
    }
    
    // Disclaimer
    resultsHTML += `
        <div class="disclaimer">
            <strong>Disclaimer:</strong> This is an AI-assisted preliminary analysis. 
            Always consult with qualified medical professionals for diagnosis and treatment decisions.
        </div>
    `;
    
    resultsHTML += '</div>';
    
    // Store analysis data for PDF generation
    window.currentAnalysisData = data;
    
    analysisResults.innerHTML = resultsHTML;
    }

    // Format AI Report for better display
    function formatAIReport(reportText) {
        if (!reportText) return '';
        
        let formattedHTML = '';
        
        // Split the report into sections
        const lines = reportText.split('\n').filter(line => line.trim() !== '');
        
        let currentSection = '';
        
        lines.forEach(line => {
            const trimmedLine = line.trim();
            
            // Check for section headers
            if (trimmedLine.includes('**FINDINGS:**') || trimmedLine.includes('FINDINGS:')) {
                currentSection = 'findings';
                formattedHTML += '<div class="summary-findings">';
                formattedHTML += '<h4>🔍 Findings</h4>';
            } else if (trimmedLine.includes('**IMPRESSION:**') || trimmedLine.includes('IMPRESSION:')) {
                if (currentSection === 'findings') formattedHTML += '</div>';
                currentSection = 'impression';
                formattedHTML += '<div class="summary-impression">';
                formattedHTML += '<h4>💡 Impression</h4>';
            } else if (trimmedLine.includes('**RECOMMENDATION:**') || trimmedLine.includes('RECOMMENDATION:')) {
                if (currentSection === 'impression') formattedHTML += '</div>';
                currentSection = 'recommendation';
                formattedHTML += '<div class="summary-recommendation">';
                formattedHTML += '<h4>📋 Recommendation</h4>';
            } else if (trimmedLine && !trimmedLine.startsWith('**') && !trimmedLine.endsWith('**')) {
                // Regular content line
                const cleanLine = trimmedLine.replace(/\*\*/g, ''); // Remove any remaining ** markers
                formattedHTML += `<p>${cleanLine}</p>`;
            }
        });
        
        // Close any open section
        if (currentSection) {
            formattedHTML += '</div>';
        }
        
        // If no sections were found, display as simple formatted text
        if (!formattedHTML) {
            formattedHTML = `<div class="summary-findings"><p>${reportText.replace(/\n/g, '<br>')}</p></div>`;
        }
        
        return formattedHTML;
    }

    // Download PDF functionality
    downloadBtn.addEventListener('click', function() {
        if (!window.currentAnalysisData) {
            alert('No analysis data available for PDF generation.');
            return;
        }

        // For now, create a simple text report that can be saved
        const { probabilities, report_text } = window.currentAnalysisData;
        
        let reportContent = 'CHEST X-RAY ANALYSIS REPORT\n';
        reportContent += '=====================================\n\n';
        
        reportContent += 'TOP OBSERVATIONS:\n';
        reportContent += '-----------------\n';
        if (probabilities) {
            Object.entries(probabilities).forEach(([condition, probability]) => {
                const percentage = (probability * 100).toFixed(1);
                reportContent += `${condition.replace('_', ' ')}: ${percentage}%\n`;
            });
        }
        
        reportContent += '\nAI SUMMARY:\n';
        reportContent += '-----------\n';
        if (report_text) {
            reportContent += report_text + '\n';
        }
        
        reportContent += '\nDISCLAIMER:\n';
        reportContent += '-----------\n';
        reportContent += 'This is an AI-assisted preliminary analysis. Always consult with qualified medical professionals for diagnosis and treatment decisions.\n';
        
        // Create and download the text file
        const blob = new Blob([reportContent], { type: 'text/plain' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `xray-analysis-report-${new Date().toISOString().split('T')[0]}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
        
        // TODO: Implement PDF generation API endpoint in backend
        // This could call a new endpoint like /generate-pdf that creates a proper PDF report
    });

    // Drag and drop functionality
    const uploadContainer = document.querySelector('.upload-container');

    uploadContainer.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadContainer.style.borderColor = 'rgba(255, 255, 255, 0.6)';
        uploadContainer.style.background = 'rgba(255, 255, 255, 0.2)';
    });

    uploadContainer.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadContainer.style.borderColor = 'rgba(255, 255, 255, 0.3)';
        uploadContainer.style.background = 'rgba(255, 255, 255, 0.1)';
    });

    uploadContainer.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadContainer.style.borderColor = 'rgba(255, 255, 255, 0.3)';
        uploadContainer.style.background = 'rgba(255, 255, 255, 0.1)';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            
            // Validate file type
            const validTypes = ['image/png', 'image/jpg', 'image/jpeg'];
            if (!validTypes.includes(file.type)) {
                alert('Please upload a PNG or JPG file.');
                return;
            }

            // Validate file size (max 10MB)
            if (file.size > 10 * 1024 * 1024) {
                alert('File size should be less than 10MB.');
                return;
            }

            // Show loading screen
            loading.style.display = 'flex';
            
            // Create file reader to display image
            const reader = new FileReader();
            reader.onload = function(e) {
                // Hide loading after a realistic delay
                setTimeout(() => {
                    // Set the uploaded image
                    xrayImage.src = e.target.result;
                    xrayImage.style.display = 'block';
                    xrayPlaceholder.style.display = 'none';
                    
                    // Hide page 1, show page 2
                    page1.style.display = 'none';
                    page2.style.display = 'block';
                    loading.style.display = 'none';
                    
                    // Call real API for analysis
                    performAnalysis(file);
                    
                }, 1500);
            };
            reader.readAsDataURL(file);
        }
    });

    // Keyboard accessibility
    uploadContainer.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            fileInput.click();
        }
    });

    // Make upload container focusable
    uploadContainer.setAttribute('tabindex', '0');
    uploadContainer.setAttribute('role', 'button');
    uploadContainer.setAttribute('aria-label', 'Upload chest X-ray scan');

    // Reset functionality (optional - can be triggered programmatically)
    function resetApp() {
        page1.style.display = 'flex';
        page2.style.display = 'none';
        fileInput.value = '';
        xrayImage.src = '';
        xrayImage.style.display = 'none';
        xrayPlaceholder.style.display = 'block';
        analysisResults.innerHTML = 'Analysis results will appear here...';
        downloadBtn.disabled = false;
        
        // Clear stored analysis data
        delete window.currentAnalysisData;
    }

// Add this to global scope for debugging/testing
window.resetApp = resetApp;

// API connectivity test function
async function testApiConnection() {
    const apiUrl = CONFIG.getApiUrl();
    console.log('🔍 Testing API connection to:', apiUrl);
    
    try {
        const response = await fetch(`${apiUrl}/`, {
            method: 'GET',
            mode: 'cors'
        });
        
        if (response.ok) {
            const data = await response.json();
            alert(`✅ API Connection Successful!\n\nAPI Response: ${JSON.stringify(data, null, 2)}`);
        } else {
            alert(`⚠️ API responded but with status: ${response.status}\n\nCheck backend deployment.`);
        }
    } catch (error) {
        alert(`❌ API Connection Failed!\n\nError: ${error.message}\n\nAPI URL: ${apiUrl}\n\nCheck:\n1. Backend is deployed and running\n2. CORS is configured\n3. API URL is correct`);
    }
}

// Initialize app and check API connectivity on load
window.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 App initialized');
    
    // Test API connection in development
    if (CONFIG.isDevelopment()) {
        console.log('🔍 Development mode - testing API connection...');
        setTimeout(() => {
            testApiConnection().catch(err => {
                console.warn('⚠️ API connection test failed:', err.message);
            });
        }, 1000);
    } else {
        console.log('🌐 Production mode - API:', CONFIG.getApiUrl());
    }
});

/*
BACKEND API INTEGRATION COMPLETE:

✅ Image Upload & Analysis: POST /analyze
   - Sends FormData with image file
   - Returns: { probabilities: {...}, heatmap_image: "base64...", report_text: "..." }

✅ Real-time Analysis Display:
   - Top 5 pathology probabilities with visual bars
   - Heatmap visualization overlay
   - AI-generated radiology report summary
   - Proper error handling for API failures

✅ Data Flow:
   1. User uploads X-ray image (drag & drop or click)
   2. Image sent to backend ML model for analysis
   3. Model returns probabilities, generates heatmap, calls Gemini API
   4. Frontend displays results in structured format
   5. User can download text report (PDF generation can be added)

⚠️ Requirements:
   - Backend server must be running on http://127.0.0.1:8000
   - CORS enabled in FastAPI backend
   - Gemini API key configured for AI reports
   - Model weights (NIHDenseNet.pth) loaded

🔧 Future Enhancements:
   - Add PDF generation endpoint to backend
   - Implement user authentication
   - Add image history/storage
   - Enhanced error handling and retry mechanisms
   - Real-time progress indicators for long analyses
*/

}); // End of DOMContentLoaded event listener
