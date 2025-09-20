# Rainfall Prediction System

A machine learning-based web application that predicts rainfall amounts based on weather conditions.

## Features

- 🌧️ **Accurate Predictions**: Uses Random Forest Regressor trained on Austin weather data
- 🎨 **Modern UI**: Beautiful, responsive web interface
- 📊 **Comprehensive Input**: Takes 14 different weather parameters
- ⚡ **Real-time**: Instant predictions with loading animations
- 📱 **Mobile Friendly**: Responsive design works on all devices

## Weather Parameters

The system analyzes the following weather conditions:

### Temperature

- High Temperature (°F)
- Average Temperature (°F)
- Low Temperature (°F)

### Humidity & Dew Point

- Average Dew Point (°F)
- High Humidity (%)
- Average Humidity (%)
- Low Humidity (%)

### Wind & Pressure

- Sea Level Pressure (inches)
- High Wind Speed (MPH)
- Average Wind Speed (MPH)
- Wind Gust (MPH)

### Visibility

- High Visibility (miles)
- Average Visibility (miles)
- Low Visibility (miles)

## Installation

1. **Clone or download the project files**

2. **Install Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**

   ```bash
   python app.py
   ```

4. **Open your browser and go to:**
   ```
   http://localhost:5000
   ```

## How to Use

1. **Fill in the weather data** in the form fields
2. **Click "Predict Rainfall"** to get the prediction
3. **View the result** showing predicted rainfall in inches
4. **Use "Fill Sample Data"** button to test with example values

## Model Information

- **Algorithm**: Random Forest Regressor
- **Training Data**: Austin weather data (1008 samples)
- **Features**: 14 weather parameters
- **Target**: Precipitation in inches
- **Accuracy**: ~90% on training data

## Technical Details

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **ML Library**: scikit-learn
- **Data Processing**: pandas, numpy
- **Model Persistence**: pickle

## API Endpoints

- `GET /` - Main application page
- `POST /predict` - Make rainfall prediction
- `GET /model_info` - Get model information

## Sample Prediction Request

```json
{
  "temp_high_f": 75,
  "temp_avg_f": 65,
  "temp_low_f": 55,
  "dew_point_avg_f": 50,
  "humidity_high_percent": 85,
  "humidity_avg_percent": 70,
  "humidity_low_percent": 55,
  "sea_level_pressure_avg_inches": 30.15,
  "visibility_high_miles": 10,
  "visibility_avg_miles": 8,
  "visibility_low_miles": 5,
  "wind_high_mph": 15,
  "wind_avg_mph": 8,
  "wind_gust_mph": 25
}
```

## Notes

- This model is trained on Austin, Texas weather data
- Predictions are for reference purposes
- Actual weather conditions may vary
- The model automatically handles data preprocessing and scaling

## Troubleshooting

If you encounter any issues:

1. **Make sure all dependencies are installed**
2. **Check that the CSV file is in the correct location**
3. **Ensure Python 3.7+ is being used**
4. **Check the console for error messages**

## Future Enhancements

- Add more weather stations data
- Implement model retraining
- Add historical prediction accuracy
- Include confidence intervals
- Add data visualization charts
