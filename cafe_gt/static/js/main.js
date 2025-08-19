document.addEventListener("DOMContentLoaded", function() {
    const baristaBtn = document.getElementById('barista-btn');
    const chatbotDiv = document.getElementById('chatbot');
    const chatInput = document.getElementById('chat-input');
    const chatbox = document.getElementById('chatbox');

    // Toggle chatbot visibility when Barista button is clicked
    if (baristaBtn) {
        baristaBtn.addEventListener('click', function() {
            chatbotDiv.classList.toggle('hidden');
        });
    }

    // Send message when Enter key is pressed
    if (chatInput) {
        chatInput.addEventListener('keydown', function(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        });
    }

    // Add product to cart
    window.addToCart = function(itemName, itemPrice) {
        let cart = JSON.parse(localStorage.getItem('cart')) || [];
        const itemIndex = cart.findIndex(item => item.name === itemName);

        if (itemIndex > -1) {
            cart[itemIndex].quantity += 1;
        } else {
            cart.push({ name: itemName, price: itemPrice, quantity: 1 });
        }
        localStorage.setItem('cart', JSON.stringify(cart));
        alert(`${itemName} added to cart!`);
        updateCartDisplay();
    };

    // Send message to chatbot
    window.sendMessage = async function() {
        const message = chatInput.value.trim();
        if (!message) return;

        displayMessage(message, 'user');
        chatInput.value = '';

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message }),
            });
            const data = await response.json();
            displayMessage(data.reply, 'bot');
        } catch (error) {
            console.error('Chatbot communication error:', error);
            displayMessage('Sorry, I cannot communicate right now.', 'bot');
        }
    };

    // Display messages in chatbox
    function displayMessage(message, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add(sender + '-message');
        messageDiv.textContent = message;
        chatbox.appendChild(messageDiv);
        chatbox.scrollTop = chatbox.scrollHeight;
    }

    // Update cart items and total price
    function updateCartDisplay() {
        const cartItemsList = document.getElementById('cart-items');
        const cartTotalSpan = document.getElementById('cart-total');
        if (cartItemsList && cartTotalSpan) {
            let cart = JSON.parse(localStorage.getItem('cart')) || [];
            cartItemsList.innerHTML = '';
            let total = 0;

            if (cart.length === 0) {
                cartItemsList.innerHTML = '<li style="text-align:center; color:#888;">Your cart is empty.</li>';
            } else {
                cart.forEach((item, index) => {
                    const listItem = document.createElement('li');
                    listItem.textContent = `${item.name} x${item.quantity} - ${(item.price * item.quantity).toFixed(2)} TL`;
                    
                    const removeButton = document.createElement('button');
                    removeButton.textContent = 'Remove';
                    removeButton.className = 'remove-from-cart-btn';
                    removeButton.onclick = () => removeFromCart(index);
                    
                    listItem.appendChild(removeButton);
                    cartItemsList.appendChild(listItem);
                    total += item.price * item.quantity;
                });
            }
            cartTotalSpan.textContent = `Total: ${total.toFixed(2)} TL`;
        }
    }

    // Remove item from cart
    window.removeFromCart = function(index) {
        let cart = JSON.parse(localStorage.getItem('cart')) || [];
        cart.splice(index, 1);
        localStorage.setItem('cart', JSON.stringify(cart));
        updateCartDisplay();
    };

    // Handle checkout form submission (on checkout.html)
    const checkoutForm = document.getElementById('checkout-form');
    if (checkoutForm) {
        checkoutForm.addEventListener('submit', async function(event) {
            event.preventDefault();

            const customerName = document.getElementById('customer-name').value.trim();
            const customerEmail = document.getElementById('customer-email').value.trim();
            const deliveryAddress = document.getElementById('delivery-address').value.trim();
            
            if (!customerName || !customerEmail || !deliveryAddress) {
                const msg = document.getElementById('checkout-message');
                msg.textContent = 'Please fill in all fields.';
                msg.style.color = 'red';
                return;
            }

            const cartItems = JSON.parse(localStorage.getItem('cart')) || [];
            const totalPriceText = document.getElementById('cart-total').textContent;
            const totalPrice = parseFloat(totalPriceText.replace('Total: ', '').replace(' TL', ''));

            const orderData = {
                customerName,
                customerEmail,
                deliveryAddress,
                cartItems,
                totalPrice
            };
            
            try {
                const response = await fetch('/checkout', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(orderData),
                });
                const result = await response.json();
                const messageDiv = document.getElementById('checkout-message');
                if (result.status === 'success') {
                    messageDiv.textContent = result.message;
                    messageDiv.style.color = 'green';
                    localStorage.removeItem('cart');
                    setTimeout(() => { window.location.href = '/'; }, 3000);
                } else {
                    messageDiv.textContent = result.message;
                    messageDiv.style.color = 'red';
                }
            } catch (error) {
                console.error('Checkout error:', error);
                const msg = document.getElementById('checkout-message');
                msg.textContent = 'Unexpected error occurred. Please try again.';
                msg.style.color = 'red';
            }
        });
    }

    // Update cart display on page load for cart.html and checkout.html
    if (window.location.pathname.includes('/cart') || window.location.pathname.includes('/checkout')) {
        updateCartDisplay();
    }
});
