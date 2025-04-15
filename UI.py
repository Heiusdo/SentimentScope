import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pandas as pd
import plotly.express as px

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the new model (cardiffnlp/twitter-roberta-base-sentiment)
model_path = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# Sentiment prediction function (adjusted for CardiffNLP’s 3-class output)
def predict_sentiment(texts):
    if isinstance(texts, str):
        texts = [texts]
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        confidences = probs.max(dim=1).values.cpu().numpy()
    
    results = []
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        # CardiffNLP mapping: 0 = Negative, 1 = Neutral, 2 = Positive
        # For binary, treat Neutral (1) as Negative
        label = "Positive" if pred == 2 else "Negative"
        results.append({"Text": texts[i], "Sentiment": label, "Confidence": conf})
    return pd.DataFrame(results)

# Streamlit interface
st.set_page_config(page_title="Thesis Sentiment Dashboard", layout="wide")
st.title("Sentiment Analysis Dashboard - HieuD's Thesis")
st.write("Binary sentiment analysis based on CardiffNLP RoBERTa - 2025")

# Sidebar: Data source options
st.sidebar.header("Data Source")
# Comment out upload option
data_source = st.sidebar.radio("Select source:", ("Manual Input", "Sample Data from latest.csv"))
# uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file (requires 'Text' column)", type=["csv", "xlsx"])

# Initialize session state
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "df_uploaded" not in st.session_state:
    st.session_state.df_uploaded = None

# 1. Input Data
st.header("1. Input Data")
input_texts = []
if data_source == "Manual Input":
    user_input = st.text_area("Enter sentences (one per line):", "I love this product\nThis service is terrible")
    input_texts = user_input.split("\n") if user_input else []
# Comment out upload handling
# elif data_source == "Upload CSV/Excel" and uploaded_file:
#     if uploaded_file.name.endswith(".csv"):
#         st.session_state.df_uploaded = pd.read_csv(uploaded_file)
#     elif uploaded_file.name.endswith(".xlsx"):
#         st.session_state.df_uploaded = pd.read_excel(uploaded_file)
#     if "Text" not in st.session_state.df_uploaded.columns:
#         st.error("File must contain a 'Text' column!")
#         st.stop()
#     input_texts = st.session_state.df_uploaded["Text"].tolist()
#     st.write("Preview of uploaded file (first 5 rows):")
#     st.dataframe(st.session_state.df_uploaded.head())
elif data_source == "Sample Data from latest.csv":
    df_sample = pd.read_csv(r"C:\Users\hieud\Documents\draft thesis\thesis\src\data\latest.csv").dropna()
    df_sample = df_sample[df_sample["Sentiment"] != 1].sample(100, random_state=42)
    input_texts = df_sample["Text"].tolist()
    st.write("Displaying 5 samples from latest.csv:")
    st.dataframe(df_sample[["Text", "Sentiment"]].head())
else:
    st.write("Please select a data source and provide input.")

# 2. Sentiment Predictions
st.header("2. Sentiment Predictions")
if st.button("Predict") and input_texts:
    st.session_state.results_df = predict_sentiment(input_texts)
    st.subheader("Results Table")
    st.dataframe(st.session_state.results_df.style.format({"Confidence": "{:.2%}"}))

    # Sentiment distribution chart
    sentiment_counts = st.session_state.results_df["Sentiment"].value_counts()
    fig_pie = px.pie(values=sentiment_counts, names=sentiment_counts.index, title="Sentiment Distribution")
    st.plotly_chart(fig_pie)

    # Highlight high-confidence predictions (>90%)
    high_conf = st.session_state.results_df[st.session_state.results_df["Confidence"] > 0.9]
    if not high_conf.empty:
        st.subheader("High-Confidence Predictions (>90%)")
        st.dataframe(high_conf.style.format({"Confidence": "{:.2%}"}))

# 3. Model Performance Metrics (New CardiffNLP Model)
st.header("3. Model Performance Metrics (CardiffNLP RoBERTa)")
col1, col2 = st.columns(2)
with col1:
    st.write("Test Accuracy: **79.10%**")
    epochs = list(range(1, 11))
    accuracy = [0.7581, 0.7923, 0.8098, 0.8259, 0.8418, 0.8569, 0.8714, 0.8811, 0.8895, 0.8957]
    loss = [0.4926, 0.4444, 0.4131, 0.3843, 0.3555, 0.3276, 0.3001, 0.2799, 0.2625, 0.2495]
    fig_curve = px.line(x=epochs, y=[accuracy, loss], labels={"x": "Epoch", "value": "Value", "variable": "Metric"},
                        title="Learning Curve", color_discrete_map={"wide_variable_0": "blue", "wide_variable_1": "red"})
    fig_curve.data[0].name = "Accuracy"
    fig_curve.data[1].name = "Loss"
    st.plotly_chart(fig_curve)

with col2:
    st.write("Training vs Test Accuracy")
    comparison = pd.DataFrame({"Phase": ["Training", "Test"], "Accuracy": [0.8957, 0.7910]})
    fig_bar = px.bar(comparison, x="Phase", y="Accuracy", text=comparison["Accuracy"].apply(lambda x: f"{x:.2%}"),
                     title="Training vs Test")
    st.plotly_chart(fig_bar)

# 4. Model Comparison (Old vs New)
st.header("4. Model Comparison (Old vs New)")
old_model_stats = {
    "epochs": list(range(1, 11)),
    "accuracy": [0.7187, 0.7678, 0.7838, 0.7974, 0.8087, 0.8211, 0.8315, 0.8399, 0.8489, 0.8531],
    "loss": [0.5376, 0.4817, 0.4566, 0.4343, 0.4129, 0.3922, 0.3734, 0.3570, 0.3404, 0.3325],
    "test_accuracy": 0.7832,
}

new_model_stats = {
    "epochs": list(range(1, 11)),
    "accuracy": [0.7581, 0.7923, 0.8098, 0.8259, 0.8418, 0.8569, 0.8714, 0.8811, 0.8895, 0.8957],
    "loss": [0.4926, 0.4444, 0.4131, 0.3843, 0.3555, 0.3276, 0.3001, 0.2799, 0.2625, 0.2495],
    "test_accuracy": 0.7910,
}

col1, col2 = st.columns(2)
with col1:
    # Accuracy comparison
    fig_acc = px.line(x=old_model_stats["epochs"], y=[old_model_stats["accuracy"], new_model_stats["accuracy"]],
                      labels={"x": "Epoch", "value": "Accuracy", "variable": "Model"},
                      title="Training Accuracy: Old vs New",
                      color_discrete_map={"wide_variable_0": "orange", "wide_variable_1": "green"})
    fig_acc.data[0].name = "Old Model (Facebook RoBERTa)"
    fig_acc.data[1].name = "New Model (CardiffNLP)"
    st.plotly_chart(fig_acc)

with col2:
    # Loss comparison
    fig_loss = px.line(x=old_model_stats["epochs"], y=[old_model_stats["loss"], new_model_stats["loss"]],
                       labels={"x": "Epoch", "value": "Loss", "variable": "Model"},
                       title="Training Loss: Old vs New",
                       color_discrete_map={"wide_variable_0": "orange", "wide_variable_1": "green"})
    fig_loss.data[0].name = "Old Model (Facebook RoBERTa)"
    fig_loss.data[1].name = "New Model (CardiffNLP)"
    st.plotly_chart(fig_loss)

# Test accuracy comparison
st.subheader("Test Accuracy Comparison")
test_comparison = pd.DataFrame({
    "Model": ["Old Model (Facebook RoBERTa)", "New Model (CardiffNLP)"],
    "Test Accuracy": [old_model_stats["test_accuracy"], new_model_stats["test_accuracy"]]
})
fig_test_bar = px.bar(test_comparison, x="Model", y="Test Accuracy",
                      text=test_comparison["Test Accuracy"].apply(lambda x: f"{x:.2%}"),
                      title="Test Accuracy: Old vs New")
st.plotly_chart(fig_test_bar)

# 5. Example Test Cases
st.header("5. Example Test Cases")
example_texts = ["I love you", "This product is amazing", "I hate this experience", "The service was terrible"]
example_results = predict_sentiment(example_texts)
st.write("Example sentences from the file:")
st.dataframe(example_results.style.format({"Confidence": "{:.2%}"}))
new_example = st.text_input("Add a new sentence to test:", "I enjoy this")
if st.button("Test New Sentence") and new_example:
    new_result = predict_sentiment(new_example)
    st.write("Result:")
    st.dataframe(new_result.style.format({"Confidence": "{:.2%}"}))

# Footer
st.write("---")
st.write("Thesis project by HieuD - 2025")