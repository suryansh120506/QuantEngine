import matplotlib.pyplot as plt

# 1. Use the values from your prediction script
actual_last_price = 1409.00  # Today's closing price
predicted_price = 1439.11    # Your model's prediction

# 2. Plotting the 'Momentum'
plt.figure(figsize=(10,6))
plt.plot([0, 1], [actual_last_price, predicted_price], marker='o', linestyle='--', color='green', label='Predicted Trend')
plt.title('Reliance.NS: 24-Hour Quant Forecast')
plt.ylabel('Price in ₹')
plt.xticks([0, 1], ['Today (Actual)', 'Tomorrow (Predicted)'])
plt.grid(True)
plt.legend()
plt.show()