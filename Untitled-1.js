document.addEventListener('DOMContentLoaded', () => {
    // Retrieve required DOM elements
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');

    // Validate that required elements exist
    if (!chatMessages) {
        console.error("Missing element with id 'chat-messages'");
        return;
    }
    if (!userInput) {
        console.error("Missing element with id 'user-input'");
        return;
    }
    if (!sendBtn) {
        console.error("Missing element with id 'send-btn'");
        return;
    }

    // Function to add a message to the chat
    const addMessage = (message, isBot = false) => {
        const messageElement = document.createElement('div');
        messageElement.className = isBot ? 'bot-message message' : 'user-message message';
        messageElement.innerHTML = message;
        chatMessages.appendChild(messageElement);
        chatMessages.scrollTop = chatMessages.scrollHeight; // Auto-scroll to the latest message
    };

    // Function to send a message
    const sendMessage = () => {
        const userText = userInput.value.trim();
        if (!userText) {
            alert("Please enter a message.");
            return;
        }

        // Add user's message to the chat
        addMessage(userText, false);
        userInput.value = ''; // Clear input field

        // Send the message to the backend
        fetch('http://127.0.0.1:5000/chat', {
            method: 'POST', // Ensure the method is POST
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: userText }), // Include the message payload
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if ('reply' in data) {
                addMessage(data.reply, true); // Display the bot's reply
            } else {
                console.error("Missing 'reply' in server response.");
                addMessage("Error: No reply from server.", true);
            }
        })
        .catch(error => {
            console.error('Fetch error:', error);
            addMessage("Error: Unable to connect to server.", true);
        });
    };

    // Event listeners for send button and "Enter" key
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault(); // Prevent default form submission
            sendMessage();
        }
    });

    // Function to navigate between sections
    const navigateSections = (visibleSectionId) => {
        const sections = ['section1', 'section2', 'section3', 'section4'];
        sections.forEach(id => {
            const section = document.getElementById(id);
            if (section) {
                section.style.display = id === visibleSectionId ? 'block' : 'none';
            } else {
                console.error(`Section with id '${id}' not found.`);
            }
        });
    };

    // Functions to expose for section navigation
    window.goToSection4 = () => navigateSections('section4');
    window.backToHome = () => navigateSections('section1');
});

    
