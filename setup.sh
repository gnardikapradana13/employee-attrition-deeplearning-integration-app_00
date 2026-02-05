#!/bin/bash
# setup.sh - Setup script for Streamlit Cloud

echo "🔧 Setting up Employee Attrition Predictor..."

# Create necessary directories
mkdir -p .streamlit
mkdir -p data
mkdir -p models

# Check if model file exists, if not create dummy
if [ ! -f "employee_attrition_nn.h5" ]; then
    echo "⚠ Model file not found. Creating dummy model for demo..."
    python -c "
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Create dummy model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(20,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save model
model.save('employee_attrition_nn.h5')
print('✅ Dummy model created')
"
fi

echo "✅ Setup complete!"