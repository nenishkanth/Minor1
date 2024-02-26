# app.py

import os
from flask import Flask, render_template, request, Markup, send_file
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib
matplotlib.use('Agg')
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__, static_url_path='/static')

# Load the model and vectorizer
df = pd.read_table(r'/Users/mirmuzammilali/Downloads/sms+spam+collection/SMSSpamCollection', sep="\t", header=None,
                   names=["label", "sms_message"])
df["label"] = df.label.map({"ham": 0, "spam": 1})

count_vector = CountVectorizer()
train = count_vector.fit_transform(df["sms_message"])
naive_bayes = MultinomialNB()
naive_bayes.fit(train, df["label"])

# Prediction history list
predictions_history = []

@app.route('/')
def home():
    return render_template('index.html', predictions=predictions_history)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sms_message = request.form['sms_message']

        try:
            # Predict SMS message
            predict_sms = naive_bayes.predict(count_vector.transform([sms_message]))
            result = "Message is ham" if predict_sms[0] == 0 else "Message is spam"

            # Sanitize user input before rendering
            sms_message = Markup.escape(sms_message)

            # Add the current prediction to the history
            predictions_history.append({"sms_message": sms_message, "prediction": result})

            # Generate and save a bar chart
            plot_path = generate_bar_chart(predictions_history)

            return render_template('index.html', result=result, sms_message=sms_message,
                                   predictions=predictions_history, plot_path=plot_path)

        except Exception as e:
            # Handle any errors that might occur during prediction
            error_message = f"An error occurred: {str(e)}"
            return render_template('index.html', result=error_message, sms_message=sms_message,
                                   predictions=predictions_history)

def generate_bar_chart(predictions_history):
    # Prepare data for the bar chart
    labels = ["Ham", "Spam"]
    ham_count = sum(1 for prediction in predictions_history if prediction["prediction"] == "Message is ham")
    spam_count = sum(1 for prediction in predictions_history if prediction["prediction"] == "Message is spam")
    values = [ham_count, spam_count]

    # Create a bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=['#2ecc71', '#e74c3c'])
    plt.title('Prediction History Distribution')
    plt.xlabel('Prediction')
    plt.ylabel('Count')

    # Save the plot to a BytesIO object
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    plt.close()

    # Encode the BytesIO object to base64
    plot_path = "data:image/png;base64," + base64.b64encode(image_stream.read()).decode('utf-8')

    return plot_path

# New route for generating and displaying the bar chart
@app.route('/plot')
def plot():
    # Prepare data for the bar chart (you can customize this based on your requirements)
    categories = ['Ham', 'Spam']
    counts = [sum(1 for prediction in predictions_history if prediction["prediction"] == "Message is ham"),
              sum(1 for prediction in predictions_history if prediction["prediction"] == "Message is spam")]

    # Create a bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(categories, counts, color=['#2ecc71', '#e74c3c'])
    plt.title('Prediction History Distribution')
    plt.xlabel('Prediction')
    plt.ylabel('Count')

    # Save the plot to a file
    plot_path = os.path.join(app.static_folder, 'plot.png')
    plt.savefig(plot_path)
    plt.close()

    # Return the rendered template with the plot path
    return render_template('plot.html', plot_path='plot.png')

if __name__ == '__main__':
    app.run(debug=True)
