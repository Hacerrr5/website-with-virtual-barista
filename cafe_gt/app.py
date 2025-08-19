# -----------------------------------------------
# app.py
# -----------------------------------------------
# Flask application for the Virtual Barista Web App
# Includes: product catalog, cart, checkout, admin panel,
# chatbot integration, email notifications, and user authentication
# -----------------------------------------------

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_cors import CORS
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import smtplib
from email.mime.text import MIMEText
import torch
import torch.nn as nn
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
import json
import os
import random

# -----------------------------------------------
# Flask app setup, CORS, Secret Key, and Database
# -----------------------------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
app.config['SECRET_KEY'] = 'cok-gizli-ve-benzersiz-anahtar'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Login manager setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# -----------------------------------------------
# Database models
# -----------------------------------------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

    def __repr__(self):
        return f"User('{self.username}')"

class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)
    price = db.Column(db.Float, nullable=False)
    image_url = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        return f"Product('{self.name}', '{self.price}')"

class Order(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    customer_name = db.Column(db.String(100), nullable=False)
    customer_email = db.Column(db.String(100), nullable=False)
    delivery_address = db.Column(db.String(200), nullable=False)
    order_details = db.Column(db.String(500), nullable=False) 
    total_price = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f"Order('{self.customer_name}', '{self.order_details}')"

# -----------------------------------------------
# User loader for Flask-Login
# -----------------------------------------------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# -----------------------------------------------
# Chatbot setup and utility functions
# -----------------------------------------------
stemmer = PorterStemmer()

def tokenize(sentence):
    """Tokenize sentence into words."""
    return nltk.word_tokenize(sentence)

def stem(word):
    """Stem a word to its root form."""
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """Create a bag-of-words array for the input sentence."""
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# Neural network architecture
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

# Chatbot class
class Chatbot:
    def __init__(self, model_path):
        self.model = None
        self.all_words = []
        self.tags = []
        if not os.path.exists(model_path):
            print(f"Error: Chatbot model not found: {model_path}")
            return
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            checkpoint = torch.load(model_path, map_location=device)
            input_size = checkpoint["input_size"]
            hidden_size = checkpoint["hidden_size"]
            output_size = checkpoint["output_size"]
            self.all_words = checkpoint["all_words"]
            self.tags = checkpoint["tags"]
            self.model = NeuralNet(input_size, hidden_size, output_size).to(device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model.eval()
            print(f"Chatbot model loaded successfully: {model_path} ({device})")
        except Exception as e:
            print(f"Error loading chatbot model: {e}")
            self.model = None

    def get_response(self, text: str) -> str:
        if self.model is None:
            return "Sorry, the chatbot model is not available right now."
        sentence = tokenize(text)
        X = bag_of_words(sentence, self.all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).float()
        output = self.model(X)
        _, predicted = torch.max(output, dim=1)
        tag = self.tags[predicted.item()]
        try:
            with open('intents.json', 'r', encoding='utf-8') as f:
                intents_data = json.load(f)
            for intent in intents_data['intents']:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])
        except FileNotFoundError:
            return "Virtual Barista: 'intents.json' not found."
        except Exception as e:
            return f"Virtual Barista: Error finding response: {e}"
        return "Virtual Barista: I didn't understand that."

# Load chatbot model
chatbot_model_full_path = os.path.join(os.path.dirname(__file__), "chatbot_model.pth")
chatbot = Chatbot(chatbot_model_full_path)

# -----------------------------------------------
# Email sending function
# -----------------------------------------------
def send_email(to, message):
    sender_email = "seninmail@gmail.com"
    sender_password = "SENİN_UYGULAMA_ŞİFREN"
    if sender_email == "seninmail@gmail.com" or sender_password == "SENİN_UYGULAMA_ŞİFREN":
        print("\nWARNING: Please update sender email and password in app.py.")
        return
    msg = MIMEText(message)
    msg["Subject"] = "New Order Notification"
    msg["From"] = sender_email
    msg["To"] = to
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to, msg.as_string())
            print(f"Email sent successfully to: {to}")
    except Exception as e:
        print(f"Email sending error: {e}")

# -----------------------------------------------
# Flask routes
# -----------------------------------------------
@app.route("/")
def home():
    products = Product.query.all()
    return render_template("index.html", products=products)

@app.route("/cart.html")
def cart_page():
    return render_template("cart.html")

@app.route("/checkout.html")
def checkout_page():
    return render_template("checkout.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    if not data or "message" not in data:
        return jsonify({"reply": "Invalid request."}), 400
    reply = chatbot.get_response(data["message"])
    return jsonify({"reply": reply})

@app.route("/checkout", methods=["POST"])
def process_checkout():
    data = request.json
    try:
        order = Order(
            customer_name=data['customerName'],
            customer_email=data['customerEmail'],
            delivery_address=data['deliveryAddress'],
            order_details=json.dumps(data['cartItems']), 
            total_price=data['totalPrice']
        )
        db.session.add(order)
        db.session.commit()
        send_email("onecat6565@gmail.com", json.dumps(data, indent=2))
        return jsonify({
            "status": "success",
            "message": "Your order has been received and saved."
        })
    except Exception as e:
        db.session.rollback()
        print(f"Error saving order: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error saving order: {str(e)}"
        }), 500

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('admin_page'))
        else:
            flash('Invalid username or password')
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route("/register", methods=["GET", "POST"])
def register():
    return render_template("register.html")

@app.route("/admin")
@login_required
def admin_page():
    if current_user.username != 'admin':
        flash('Unauthorized access.')
        return redirect(url_for('home'))
    
    orders = Order.query.order_by(Order.id.desc()).all()
    products = Product.query.order_by(Product.id.desc()).all()
    
    # Parse order details from JSON string
    for order in orders:
        order.order_details_parsed = json.loads(order.order_details)

    return render_template("admin.html", orders=orders, products=products)

@app.route("/admin/add_product", methods=["POST"])
@login_required
def add_product():
    if current_user.username != 'admin':
        flash('Unauthorized action.')
        return redirect(url_for('home'))

    name = request.form.get("name")
    price = request.form.get("price")
    image_url = request.form.get("image_url")

    if not all([name, price, image_url]):
        flash('All fields are required.', 'error')
        return redirect(url_for('admin_page'))
    
    try:
        price = float(price)
        if Product.query.filter_by(name=name).first():
            flash(f"Product '{name}' already exists.", 'error')
            return redirect(url_for('admin_page'))

        new_product = Product(name=name, price=price, image_url=image_url)
        db.session.add(new_product)
        db.session.commit()
        flash(f"Product '{name}' added successfully!")
    except ValueError:
        flash('Price must be a valid number.', 'error')
    except Exception as e:
        db.session.rollback()
        flash(f'Error adding product: {str(e)}', 'error')
    return redirect(url_for('admin_page'))

@app.route("/admin/delete_product/<int:product_id>", methods=["POST"])
@login_required
def delete_product(product_id):
    if current_user.username != 'admin':
        flash('Unauthorized action.')
        return redirect(url_for('home'))
    
    product = Product.query.get_or_404(product_id)
    try:
        db.session.delete(product)
        db.session.commit()
        flash(f"Product '{product.name}' deleted successfully.")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting product: {str(e)}", 'error')
    return redirect(url_for('admin_page'))

# -----------------------------------------------
# Main
# -----------------------------------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        
        # Create default admin user
        if not User.query.filter_by(username='admin').first():
            admin_user = User(username='admin', password='123')
            db.session.add(admin_user)
            db.session.commit()
            print("Default 'admin' user created (password: 123).")

        # Add sample products
        if not Product.query.first():
            sample_products = [
                Product(name="Americano", price=30.0, image_url="https://images.pexels.com/photos/27566699/pexels-photo-27566699.jpeg"),
                Product(name="Mocha", price=40.0, image_url="https://images.pexels.com/photos/63283/pexels-photo-63283.jpeg"),
                Product(name="Frappe", price=35.0, image_url="https://images.pexels.com/photos/214333/pexels-photo-214333.jpeg"),
                Product(name="Latte", price=40.0, image_url="https://images.pexels.com/photos/312418/pexels-photo-312418.jpeg"),
                Product(name="Cappuccino", price=35.0, image_url="https://images.pexels.com/photos/3879495/pexels-photo-3879495.jpeg"),
                Product(name="Espresso", price=30.0, image_url="https://images.pexels.com/photos/3704460/pexels-photo-3704460.jpeg"),
                Product(name="Iced Americano", price=35.0, image_url="https://images.pexels.com/photos/33018008/pexels-photo-33018008.jpeg"),
                Product(name="Cold Brew", price=40.0, image_url="https://images.pexels.com/photos/2067404/pexels-photo-2067404.jpeg"),
                Product(name="Filtre Kahve", price=30.0, image_url="https://images.pexels.com/photos/10137977/pexels-photo-10137977.jpeg"),
                Product(name="Espresso Macchiato", price=35.0, image_url="https://images.pexels.com/photos/6687002/pexels-photo-6687002.jpeg"),
                Product(name="Duble Espresso", price=38.0, image_url="https://images.pexels.com/photos/3304449/pexels-photo-3304449.jpeg"),
                Product(name="Affogato", price=45.0, image_url="https://images.pexels.com/photos/5621453/pexels-photo-5621453.jpeg")
            ]
            db.session.bulk_save_objects(sample_products)
            db.session.commit()
            print("Sample products added to the database.")
            
    app.run(debug=True, port=5000)
