// Application Data
const applicationData = {
    performance_data: {
        "RL Agent (Q-Learning)": 0.740,
        "Naive Bayes": 0.750,
        "Logistic Regression": 0.667,
        "Random Forest": 0.667,
        "SVM": 0.667
    },
    training_progress: {
        episodes: [0, 10, 20, 30, 40, 50],
        accuracy: [0.475, 0.713, 0.743, 0.733, 0.737, 0.740],
        rewards: [-2, 17, 19.4, 18.6, 19.0, 20.2]
    },
    sample_emails: {
        spam: [
            "Congratulations! You've won $1000! Click here to claim now!",
            "FREE MONEY! Win big cash prizes! Act now!",
            "URGENT: Your account will be closed. Click to verify immediately!",
            "Amazing offer! Buy one get one free! Limited time only!",
            "You are a winner! Claim your lottery prize now!"
        ],
        ham: [
            "Hi John, let's meet for lunch tomorrow at 12pm",
            "Thank you for your purchase. Your order has been confirmed",
            "Meeting scheduled for Monday at 3pm in conference room A",
            "Your monthly statement is now available online",
            "Happy birthday! Hope you have a wonderful day"
        ]
    },
    spam_keywords: ["free", "money", "win", "prize", "urgent", "click", "claim", "offer", "congratulations", "winner"],
    ham_keywords: ["meeting", "lunch", "thank", "order", "birthday", "project", "flight", "team", "statement", "subscription"]
};

// Simulated RL Agent for Classification
class SpamClassifierRL {
    constructor() {
        this.spamKeywords = applicationData.spam_keywords;
        this.hamKeywords = applicationData.ham_keywords;
        this.qTable = this.initializeQTable();
    }

    initializeQTable() {
        // Simulated Q-table with learned values
        return {
            spam_features: { classify_spam: 0.85, classify_ham: 0.15 },
            ham_features: { classify_spam: 0.25, classify_ham: 0.75 },
            mixed_features: { classify_spam: 0.6, classify_ham: 0.4 },
            neutral_features: { classify_spam: 0.5, classify_ham: 0.5 }
        };
    }

    extractFeatures(emailText) {
        const text = emailText.toLowerCase();
        const words = text.split(/\s+/);
        
        const spamFeatures = this.spamKeywords.filter(keyword => 
            text.includes(keyword.toLowerCase())
        );
        
        const hamFeatures = this.hamKeywords.filter(keyword => 
            text.includes(keyword.toLowerCase())
        );

        // Calculate TF-IDF like features
        const exclamationCount = (text.match(/!/g) || []).length;
        const capsRatio = (emailText.match(/[A-Z]/g) || []).length / emailText.length;
        const moneyMentions = (text.match(/\$|\bmoney\b|\bcash\b|\bfree\b/g) || []).length;
        
        return {
            spamFeatures,
            hamFeatures,
            exclamationCount,
            capsRatio,
            moneyMentions,
            wordCount: words.length
        };
    }

    getState(features) {
        const spamScore = features.spamFeatures.length + features.exclamationCount * 0.5 + 
                         features.capsRatio * 5 + features.moneyMentions;
        const hamScore = features.hamFeatures.length;

        if (spamScore > hamScore + 2) return 'spam_features';
        if (hamScore > spamScore + 1) return 'ham_features';
        if (Math.abs(spamScore - hamScore) <= 1) return 'mixed_features';
        return 'neutral_features';
    }

    classify(emailText) {
        const features = this.extractFeatures(emailText);
        const state = this.getState(features);
        const qValues = this.qTable[state];
        
        // Add some randomness to simulate real RL behavior
        const noise = (Math.random() - 0.5) * 0.1;
        const spamProb = Math.max(0, Math.min(1, qValues.classify_spam + noise));
        
        const isSpam = spamProb > 0.5;
        const confidence = Math.abs(spamProb - 0.5) * 2;
        
        return {
            classification: isSpam ? 'spam' : 'ham',
            confidence: Math.round(confidence * 100),
            features: features,
            probabilities: {
                spam: Math.round(spamProb * 100),
                ham: Math.round((1 - spamProb) * 100)
            }
        };
    }
}

// Initialize the classifier
const classifier = new SpamClassifierRL();
let performanceChart = null;

// DOM Elements
let emailInput, classifyBtn, btnText, btnLoader, resultsSection;
let resultClassification, confidenceFill, confidenceText, detectedFeatures;
let loadSpamBtn, loadHamBtn, spamExamples, hamExamples;

// Initialize DOM elements after page loads
function initializeDOMElements() {
    emailInput = document.getElementById('email-input');
    classifyBtn = document.getElementById('classify-btn');
    btnText = document.querySelector('.btn-text');
    btnLoader = document.querySelector('.btn-loader');
    resultsSection = document.getElementById('results-section');
    resultClassification = document.getElementById('result-classification');
    confidenceFill = document.getElementById('confidence-fill');
    confidenceText = document.getElementById('confidence-text');
    detectedFeatures = document.getElementById('detected-features');
    loadSpamBtn = document.getElementById('load-spam-btn');
    loadHamBtn = document.getElementById('load-ham-btn');
    spamExamples = document.getElementById('spam-examples');
    hamExamples = document.getElementById('ham-examples');
}

// Initialize Application
function initializeApp() {
    initializeDOMElements();
    setupEventListeners();
    renderExampleEmails();
    renderPerformanceChart();
}

// Setup Event Listeners
function setupEventListeners() {
    if (classifyBtn) {
        classifyBtn.addEventListener('click', classifyEmail);
    }
    
    if (loadSpamBtn) {
        loadSpamBtn.addEventListener('click', () => loadSampleEmail('spam'));
    }
    
    if (loadHamBtn) {
        loadHamBtn.addEventListener('click', () => loadSampleEmail('ham'));
    }

    // Auto-resize textarea
    if (emailInput) {
        emailInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 400) + 'px';
        });
    }

    // Keyboard Shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + Enter to classify
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            if (classifyBtn && !classifyBtn.disabled && emailInput && emailInput.value.trim()) {
                classifyEmail();
            }
        }
        
        // Escape to clear input
        if (e.key === 'Escape' && document.activeElement === emailInput) {
            emailInput.value = '';
            if (resultsSection) {
                resultsSection.classList.add('hidden');
            }
        }
    });
}

// Email Classification Function
function classifyEmail() {
    if (!emailInput || !emailInput.value.trim()) {
        alert('Please enter an email to classify');
        return;
    }

    const emailText = emailInput.value.trim();
    
    // Show loading state
    showLoadingState();
    
    // Simulate processing time with setTimeout
    setTimeout(() => {
        try {
            const result = classifier.classify(emailText);
            displayResults(result);
        } catch (error) {
            console.error('Classification error:', error);
            alert('An error occurred during classification. Please try again.');
        } finally {
            hideLoadingState();
        }
    }, 1500);
}

// Display Classification Results
function displayResults(result) {
    if (!resultsSection) return;
    
    // Show results section
    resultsSection.classList.remove('hidden');
    resultsSection.classList.add('fade-in');
    
    // Update classification result
    if (resultClassification) {
        resultClassification.textContent = result.classification.toUpperCase();
        resultClassification.className = `status status--${result.classification}`;
    }
    
    // Update confidence bar
    if (confidenceFill && confidenceText) {
        confidenceFill.style.width = `${result.confidence}%`;
        confidenceText.textContent = `${result.confidence}%`;
    }
    
    // Update detected features
    renderDetectedFeatures(result.features, result.classification);
    
    // Scroll to results
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

// Render Detected Features
function renderDetectedFeatures(features, classification) {
    if (!detectedFeatures) return;
    
    detectedFeatures.innerHTML = '';
    
    // Combine all detected features
    const allFeatures = [];
    
    // Add spam features
    features.spamFeatures.forEach(feature => {
        allFeatures.push({ name: feature, type: 'spam' });
    });
    
    // Add ham features
    features.hamFeatures.forEach(feature => {
        allFeatures.push({ name: feature, type: 'ham' });
    });
    
    // Add additional features
    if (features.exclamationCount > 2) {
        allFeatures.push({ name: `${features.exclamationCount} exclamations`, type: 'spam' });
    }
    
    if (features.capsRatio > 0.3) {
        allFeatures.push({ name: 'high caps ratio', type: 'spam' });
    }
    
    if (features.moneyMentions > 0) {
        allFeatures.push({ name: 'money mentions', type: 'spam' });
    }
    
    if (allFeatures.length === 0) {
        detectedFeatures.innerHTML = '<span class="feature-tag">No significant features detected</span>';
        return;
    }
    
    // Render feature tags
    allFeatures.forEach(feature => {
        const tag = document.createElement('span');
        tag.className = `feature-tag feature-tag--${feature.type}`;
        tag.textContent = feature.name;
        detectedFeatures.appendChild(tag);
    });
}

// Load Sample Email
function loadSampleEmail(type) {
    if (!emailInput) return;
    
    const samples = applicationData.sample_emails[type];
    const randomSample = samples[Math.floor(Math.random() * samples.length)];
    emailInput.value = randomSample;
    
    // Trigger input event for auto-resize
    emailInput.dispatchEvent(new Event('input'));
    
    // Scroll to input and focus
    setTimeout(() => {
        emailInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
        emailInput.focus();
    }, 100);
}

// Render Example Emails
function renderExampleEmails() {
    // Render spam examples
    if (spamExamples) {
        spamExamples.innerHTML = '';
        applicationData.sample_emails.spam.forEach(email => {
            const item = createExampleItem(email, 'spam');
            spamExamples.appendChild(item);
        });
    }
    
    // Render ham examples
    if (hamExamples) {
        hamExamples.innerHTML = '';
        applicationData.sample_emails.ham.forEach(email => {
            const item = createExampleItem(email, 'ham');
            hamExamples.appendChild(item);
        });
    }
}

// Create Example Item
function createExampleItem(email, type) {
    const item = document.createElement('div');
    item.className = `example-item example-item--${type}`;
    item.textContent = email;
    
    item.addEventListener('click', () => {
        if (emailInput) {
            emailInput.value = email;
            // Trigger input event for auto-resize
            emailInput.dispatchEvent(new Event('input'));
            
            // Scroll to input and focus
            setTimeout(() => {
                emailInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
                emailInput.focus();
            }, 100);
        }
    });
    
    return item;
}

// Render Performance Comparison Chart
function renderPerformanceChart() {
    const ctx = document.getElementById('comparison-chart');
    if (!ctx) return;
    
    const labels = Object.keys(applicationData.performance_data);
    const data = Object.values(applicationData.performance_data).map(val => val * 100);
    
    // Destroy existing chart if it exists
    if (performanceChart) {
        performanceChart.destroy();
    }
    
    performanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Accuracy (%)',
                data: data,
                backgroundColor: [
                    '#1FB8CD', // RL Agent - primary color
                    '#FFC185', // Naive Bayes
                    '#B4413C', // Logistic Regression
                    '#ECEBD5', // Random Forest
                    '#5D878F'  // SVM
                ],
                borderColor: [
                    '#1FB8CD',
                    '#FFC185', 
                    '#B4413C',
                    '#ECEBD5',
                    '#5D878F'
                ],
                borderWidth: 2,
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Accuracy: ${context.parsed.y.toFixed(1)}%`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    grid: {
                        color: 'rgba(0,0,0,0.1)'
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 0
                    },
                    grid: {
                        display: false
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeOutQuart'
            }
        }
    });
}

// Loading State Management
function showLoadingState() {
    if (btnText) btnText.classList.add('hidden');
    if (btnLoader) btnLoader.classList.remove('hidden');
    if (classifyBtn) {
        classifyBtn.disabled = true;
        classifyBtn.classList.add('loading');
    }
}

function hideLoadingState() {
    if (btnText) btnText.classList.remove('hidden');
    if (btnLoader) btnLoader.classList.add('hidden');
    if (classifyBtn) {
        classifyBtn.disabled = false;
        classifyBtn.classList.remove('loading');
    }
}

// Performance monitoring
let classificationCount = 0;
let correctPredictions = 0;

// Track user feedback (simulated)
function trackClassification(result) {
    classificationCount++;
    
    // Simulate accuracy based on our model's performance
    const accuracy = applicationData.performance_data["RL Agent (Q-Learning)"];
    if (Math.random() < accuracy) {
        correctPredictions++;
    }
    
    // Update performance metrics in real-time (could be displayed)
    const currentAccuracy = (correctPredictions / classificationCount) * 100;
    console.log(`Current Session Accuracy: ${currentAccuracy.toFixed(1)}%`);
}

// Feature highlighting in text (bonus feature)
function highlightFeaturesInText(text, features) {
    let highlightedText = text;
    
    // Highlight spam keywords
    features.spamFeatures.forEach(keyword => {
        const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
        highlightedText = highlightedText.replace(regex, `<span class="feature-highlight text-spam">$&</span>`);
    });
    
    // Highlight ham keywords  
    features.hamFeatures.forEach(keyword => {
        const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
        highlightedText = highlightedText.replace(regex, `<span class="feature-highlight text-ham">$&</span>`);
    });
    
    return highlightedText;
}

// Initialize everything when DOM is ready
document.addEventListener('DOMContentLoaded', initializeApp);

// Export for potential testing
window.SpamClassifier = {
    classifier,
    classifyEmail,
    applicationData
};