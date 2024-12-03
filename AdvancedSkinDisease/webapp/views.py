import cv2
import numpy as np
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login as auth_login, logout
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from joblib import load
from sklearn.preprocessing import RobustScaler
import logging
from .forms import CustomUserCreationForm
from django import forms
from django.http import HttpResponseRedirect  # Import for redirection
from django.urls import reverse

# Configure logging
logger = logging.getLogger(__name__)

# Load the model and scaler
model = load('voting_ensemble_skin_disease_model.joblib')
scaler = load('scaler.joblib')

# Disease names and corresponding diagnoses
skin_disease_names = [
    'Cellulitis',
    'Impetigo',
    'Athlete Foot',
    'Nail Fungus',
    'Ringworm',
    'Cutaneous Larva Migrans',
    'Chickenpox',
    'Shingles'
]

diagnosis = [
    'Treatment for cellulitis usually involves antibiotics, and in most cases, you should start to feel better within 7 to 10 days.',
                         'Impetigo is treated with prescription mupirocin antibiotic ointment or cream applied directly to the sores two to three times a day for five to 10 days.',
                         'Athlete foot is a fungal infection that can be treated with antifungal medications and by keeping feet clean and dry.',
                         'Nail fungus, also known as onychomycosis, is a common infection that affects the fingernails or toenails. Diagnosis typically involves a visual examination of the nail, along with a scraping or clipping of the affected nail to examine for fungal elements.',
                         'Ringworm is a fungal infection that affects the skin, hair, and nails. It is not actually caused by a worm, but rather by a type of fungus called a dermatophyte.',
                         'Cutaneous Larva Migrans (CLM) is a skin condition caused by the larvae of certain hookworms. The diagnosis typically involves a physical examination and medical history, with a focus on exposure to contaminated soil.',
                         'The diagnosis and cure for Chickenpox typically involve a combination of self-care, over-the-counter medications, and in some cases, antiviral prescriptions.',
                         'Shingles is caused by the varicella-zoster virus, the same virus that causes chickenpox. There are treatments for shingles symptoms, but there is no cure. There are vaccines against shingles and postherpetic neuralgia.'

]

def home(request):
    return render(request, 'index.html')

def login(request):
    if request.user.is_authenticated:
        return redirect('/profile')
    
    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user:
            auth_login(request, user)
            return redirect('/profile')
        else:
            return render(request, 'login.html', {
                'form': AuthenticationForm(),
                'msg': 'Invalid username or password.'
            })
    return render(request, 'login.html', {'form': AuthenticationForm()})

class CustomUserCreationForm(UserCreationForm):
    class Meta(UserCreationForm.Meta):
        fields = ['username', 'password1', 'password2']
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            field.error_messages = {
                'required': f'{field.label} is required.',
                'invalid': f'Enter a valid {field.label.lower()}.',
            }
        # Additional customization for specific fields
        self.fields['password1'].error_messages.update({
            'required': 'Password cannot be empty.',
            'invalid': 'Password is invalid. Please follow the guidelines.',
        })
        self.fields['password2'].error_messages.update({
            'required': 'Please confirm your password.',
            'password_mismatch': 'The two passwords do not match.',
        })

def register(request):
    if request.user.is_authenticated:
        return redirect('/')
    
    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return HttpResponseRedirect(reverse('login'))
        else:
            return render(request, 'register.html', {
                'form': form,
                'msg': 'Please correct the errors below.',
            })
    return render(request, 'register.html', {'form': UserCreationForm()})

def extract_features(img):
    """Extract features for prediction."""
    # Resize image
    img = cv2.resize(img, (128, 128))
    features = []

    # HOG Features
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9
    )
    hog_features = hog.compute(equalized)
    features.extend(hog_features.flatten())

    # Color Histograms (RGB and HSV)
    for color in range(3):
        hist = cv2.calcHist([img], [color], None, [256], [0, 256])
        features.extend(hist.flatten())

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for channel in range(3):
        hist = cv2.calcHist([hsv], [channel], None, [256], [0, 256])
        features.extend(hist.flatten())

    return np.array(features)

def profile(request):
    if request.method == "POST" and request.FILES.get('uploadImage'):
        # Get the uploaded image
        img_file = request.FILES['uploadImage']
        fs = FileSystemStorage()

        # Save the file to the server
        filename = fs.save(str(img_file.name), img_file)
        img_url = fs.url(filename)
        img_path = fs.path(filename)

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            return render(request, 'profile.html', {'error': 'Failed to load image'})

        try:
            # Extract and scale features
            features = extract_features(img).reshape(1, -1)
            features = scaler.transform(features)
            logger.info(f"Extracted features shape: {features.shape}")

            # Predict disease
            predict = model.predict(features)[0]
            logger.info(f"Model Prediction: {predict}")

            # Map prediction to disease and diagnosis
            if 0 <= predict < len(skin_disease_names):
                result1 = skin_disease_names[predict]
                result2 = diagnosis[predict]
            else:
                result1 = "Unknown Disease"
                result2 = "No diagnosis available."

            return render(request, 'profile.html', {
                'img': img_url,
                'obj1': result1,
                'obj2': result2
            })
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return render(request, 'profile.html', {'error': f'Prediction error: {str(e)}'})

    return render(request, 'profile.html')

def logout_view(request):
    logout(request)
    return redirect('/')
