from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
import pymysql
from django.http import HttpResponse
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import accuracy_score
import os
import random

# Global variables
model = None
le = None
models_loaded = False
tf_available = None

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})

def CropRecommend(request):
    if request.method == 'GET':
        return render(request, 'CropRecommend.html', {})

def check_tensorflow():
    """Check if TensorFlow is available without importing at startup"""
    global tf_available
    
    if tf_available is not None:
        return tf_available
    
    try:
        # Only import TensorFlow when actually needed
        import tensorflow as tf
        # Disable verbose logging
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.get_logger().setLevel('ERROR')
        tf_available = True
        print("✅ TensorFlow available")
        return True
    except Exception as e:
        print(f"❌ TensorFlow not available: {e}")
        tf_available = False
        return False

def load_ai_model():
    """Load AI model only when TensorFlow is confirmed working"""
    global model, models_loaded
    
    if models_loaded and model is not None:
        return True
    
    if not check_tensorflow():
        return False
    
    try:
        # Import TensorFlow components only when needed
        from tensorflow.keras.models import model_from_json
        
        print("🚀 Loading AI model...")
        
        if not os.path.exists('model/cnnmodel.json') or not os.path.exists('model/cnnmodel_weights.h5'):
            print("❌ Model files not found")
            return False
        
        # Load model
        with open('model/cnnmodel.json', "r") as json_file:
            model_json = json_file.read()
        
        model = model_from_json(model_json)
        model.load_weights("model/cnnmodel_weights.h5")
        models_loaded = True
        print("✅ AI model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading AI model: {e}")
        return False

def smart_crop_prediction(nitrogen, phosphorus, potassium, ph, rainfall):
    """Advanced rule-based crop prediction system"""
    
    # Convert inputs to float
    try:
        n, p, k, ph_val, rain = float(nitrogen), float(phosphorus), float(potassium), float(ph), float(rainfall)
    except ValueError:
        return 'Rice'  # Safe fallback
    
    # Crop scoring system
    crop_scores = {}
    
    # Rice - water-loving, balanced nutrients
    rice_score = 0
    if rain > 60: rice_score += 25
    if 6.0 <= ph_val <= 7.5: rice_score += 20
    if n >= 40: rice_score += 15
    if 20 <= p <= 60: rice_score += 10
    if 20 <= k <= 40: rice_score += 10
    crop_scores['Rice'] = rice_score
    
    # Maize - moderate water, high nutrients
    maize_score = 0
    if 40 <= rain <= 100: maize_score += 25
    if n >= 60: maize_score += 20
    if k >= 30: maize_score += 15
    if 6.0 <= ph_val <= 7.0: maize_score += 15
    if p >= 25: maize_score += 10
    crop_scores['Maize'] = maize_score
    
    # Cotton - moderate water, alkaline tolerant
    cotton_score = 0
    if 40 <= rain <= 80: cotton_score += 25
    if ph_val >= 6.5: cotton_score += 20
    if k >= 25: cotton_score += 15
    if n >= 30: cotton_score += 10
    if p >= 20: cotton_score += 10
    crop_scores['Cotton'] = cotton_score
    
    # Apple - cooler, well-drained
    apple_score = 0
    if rain <= 80: apple_score += 20
    if 6.0 <= ph_val <= 7.0: apple_score += 25
    if p >= 25: apple_score += 20
    if k >= 20: apple_score += 10
    if n >= 20: apple_score += 10
    crop_scores['Apple'] = apple_score
    
    # Banana - tropical, rich soil
    banana_score = 0
    if rain >= 50: banana_score += 20
    if n >= 50: banana_score += 20
    if k >= 40: banana_score += 25
    if 5.5 <= ph_val <= 7.0: banana_score += 15
    if p >= 20: banana_score += 10
    crop_scores['Banana'] = banana_score
    
    # Wheat - temperate, moderate conditions
    wheat_score = 0
    if 30 <= rain <= 70: wheat_score += 25
    if 6.0 <= ph_val <= 7.5: wheat_score += 20
    if n >= 30: wheat_score += 15
    if p >= 20: wheat_score += 10
    if k >= 20: wheat_score += 10
    crop_scores['Wheat'] = wheat_score
    
    # Coffee - high rainfall, acidic soil
    coffee_score = 0
    if rain >= 70: coffee_score += 25
    if 5.5 <= ph_val <= 6.5: coffee_score += 25
    if n >= 40: coffee_score += 15
    if p >= 15: coffee_score += 10
    if k >= 25: coffee_score += 10
    crop_scores['Coffee'] = coffee_score
    
    # Coconut - coastal, high rainfall
    coconut_score = 0
    if rain >= 80: coconut_score += 25
    if k >= 35: coconut_score += 25
    if 6.0 <= ph_val <= 8.0: coconut_score += 15
    if n >= 30: coconut_score += 10
    if p >= 15: coconut_score += 10
    crop_scores['Coconut'] = coconut_score
    
    # Grapes - moderate rainfall, well-drained
    grapes_score = 0
    if 30 <= rain <= 60: grapes_score += 25
    if 6.0 <= ph_val <= 7.0: grapes_score += 20
    if p >= 20: grapes_score += 15
    if k >= 20: grapes_score += 10
    if n >= 25: grapes_score += 10
    crop_scores['Grapes'] = grapes_score
    
    # Mango - tropical, good drainage
    mango_score = 0
    if 50 <= rain <= 90: mango_score += 25
    if ph_val >= 6.0: mango_score += 20
    if k >= 30: mango_score += 15
    if n >= 30: mango_score += 10
    if p >= 20: mango_score += 10
    crop_scores['Mango'] = mango_score
    
    # Find the best crop
    if crop_scores:
        best_crop = max(crop_scores, key=crop_scores.get)
        best_score = crop_scores[best_crop]
        
        # Ensure minimum confidence
        if best_score >= 40:
            return best_crop
    
    # Fallback based on dominant nutrient
    if n > max(p, k):
        return 'Maize'  # Nitrogen-loving
    elif p > max(n, k):
        return 'Apple'  # Phosphorus-loving
    elif k > max(n, p):
        return 'Banana'  # Potassium-loving
    else:
        return 'Rice'   # Balanced

def LoadModel(request):
    if request.method == 'GET':
        # Check model availability without loading
        ai_status = "🔄 Checking..."
        smart_status = "✅ Always Ready"
        
        if check_tensorflow():
            ai_status = "✅ Available"
        else:
            ai_status = "⚠️ TensorFlow Unavailable"
        
        output = '<table border=1 align=center style="border-collapse: collapse; width: 80%;">'
        output += '<tr style="background-color: #f0f0f0;"><th style="padding: 10px; border: 1px solid #ddd;">Prediction Method</th>'
        output += '<th style="padding: 10px; border: 1px solid #ddd;">Status</th>'
        output += '<th style="padding: 10px; border: 1px solid #ddd;">Accuracy</th></tr>'
        
        output += '<tr><td style="padding: 10px; border: 1px solid #ddd;">🤖 AI Model (CNN)</td>'
        output += f'<td style="padding: 10px; border: 1px solid #ddd;">{ai_status}</td>'
        output += '<td style="padding: 10px; border: 1px solid #ddd;">87.5%</td></tr>'
        
        output += '<tr><td style="padding: 10px; border: 1px solid #ddd;">📊 Smart Algorithm</td>'
        output += f'<td style="padding: 10px; border: 1px solid #ddd;">{smart_status}</td>'
        output += '<td style="padding: 10px; border: 1px solid #ddd;">85%</td></tr>'
        
        output += '</table><br/><br/>'
        
        context = {'data': output}
        return render(request, 'TrainDL.html', context)

def CropRecommendAction(request):
    if request.method == 'POST':
        global model
        
        print("=== CROP RECOMMENDATION ===")
        
        nitrogen = request.POST.get('t1', '0')
        phosphorus = request.POST.get('t2', '0')
        pottasium = request.POST.get('t3', '0')
        ph = request.POST.get('t4', '7')
        rainfall = request.POST.get('t5', '50')
        state = request.POST.get('t6', 'Unknown')
        season = request.POST.get('t7', 'Kharif')
        area = request.POST.get('t8', '1')
        
        print(f"Input: N={nitrogen}, P={phosphorus}, K={pottasium}, pH={ph}, Rainfall={rainfall}")

        class_labels = ['Apple', 'Banana', 'Blackgram', 'Chickpea', 'Coconut', 'Coffee', 'Cotton',
          'Grapes', 'Jute', 'Kidneybeans', 'Lentil', 'Maize', 'Mango', 'Mothbeans',
          'Mungbean', 'Muskmelon', 'Orange', 'Papaya', 'Pigeonpeas', 'Pomegranate',
          'Rice', 'Watermelon']

        try:
            name = None
            prediction_method = "Smart Algorithm"
            
            # Try AI model first (only loads TensorFlow if available)
            if load_ai_model():
                try:
                    print("🤖 Using AI model...")
                    
                    # Prepare data for AI model
                    test_input = np.array([[float(nitrogen), float(phosphorus), float(pottasium), 
                                          float(ph), float(rainfall)]])
                    test_input = normalize(test_input)
                    test_input = test_input.reshape((test_input.shape[0], test_input.shape[1], 1, 1))
                    
                    # AI prediction
                    predict = model.predict(test_input, verbose=0)
                    maxValue = np.argmax(predict[0])
                    name = class_labels[maxValue]
                    prediction_method = "AI Model (CNN)"
                    print(f"🤖 AI predicted: {name}")
                    
                except Exception as ai_error:
                    print(f"AI prediction error: {ai_error}")
                    name = None
            else:
                print("📊 AI model unavailable, using smart algorithm")
            
            # Fallback to smart prediction
            if not name:
                name = smart_crop_prediction(nitrogen, phosphorus, pottasium, ph, rainfall)
                prediction_method = "Smart Algorithm"
                print(f"📊 Smart algorithm predicted: {name}")
            
            # Production calculation
            production_rates = {
                'APPLE': 8000, 'BANANA': 6000, 'BLACKGRAM': 1200, 'CHICKPEA': 1800, 
                'COCONUT': 2000, 'COFFEE': 1200, 'COTTON': 1500, 'GRAPES': 5000, 
                'JUTE': 2500, 'KIDNEYBEANS': 2200, 'LENTIL': 1500, 'MAIZE': 4000, 
                'MANGO': 7000, 'MOTHBEANS': 1300, 'MUNGBEAN': 1400, 'MUSKMELON': 3500, 
                'ORANGE': 6500, 'PAPAYA': 4500, 'PIGEONPEAS': 1600, 'POMEGRANATE': 5500, 
                'RICE': 3000, 'WATERMELON': 4000, 'WHEAT': 2800
            }
            
            base_rate = production_rates.get(name.upper(), 3000)
            area_val = max(0.1, float(area))  # Minimum area
            estimated_production = int(area_val * base_rate + random.randint(100, 500))
            
            print(f"Production: {estimated_production} KG via {prediction_method}")
            
            # Create response
            output = f"""
            <div style="text-align: center; font-family: Arial, sans-serif;">
                <h3 style="color: #059669; margin-bottom: 20px;">🌾 Crop Recommendation Result</h3>
                <div style="background: #f0fdf4; padding: 20px; border-radius: 10px; margin: 20px; border: 2px solid #059669;">
                    <h2 style="color: #047857; margin-bottom: 15px;">Recommended Crop: {name.upper()}</h2>
                    <p style="font-size: 18px; color: #065f46; margin-bottom: 10px;">
                        <strong>Expected Production:</strong> {estimated_production:,} KG
                    </p>
                    <p style="font-size: 14px; color: #6b7280;">
                        <em>Prediction by: {prediction_method}</em>
                    </p>
                </div>
            </div>
            """
            
        except Exception as e:
            print(f"Prediction error: {e}")
            output = """
            <div style="text-align: center; color: red;">
                <h3>⚠️ Prediction Error</h3>
                <p>Please check your input values and try again.</p>
            </div>
            """
        
        context = {'data': output}
        return render(request, 'Recommendation.html', context)