import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pandas as pd
import plotly.express as px
import re

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the new model (cardiffnlp/twitter-roberta-base-sent Bermuda)
model_path = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

def preprocess_data(df):
    """
    Preprocess the input DataFrame to ensure it's suitable for sentiment analysis.
    - Identify the text column by checking common names or analyzing content.
    - Keep only rows with valid text in the selected column.
    - Clean text (remove extra spaces, normalize).
    - Drop unnamed or empty columns.
    """
    # Log raw columns for debugging
    st.write("Raw DataFrame columns:", df.columns.tolist())

    # Drop unnamed columns (e.g., 'Unnamed: 13', 'Unnamed: 14')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed', na=False)]

    # Define a broader list of common text column names (case-insensitive)
    common_text_columns = [
        'text', 'content', 'message', 'comment', 'review', 'feedback', 'description',
        'commenttext', 'reviewtext', 'post', 'tweet', 'status', 'body', 'note', 'summary'
    ]

    # Initialize text column
    text_column = None
    selected_column = st.session_state.get('selected_column', None)

    # Step 1: Check for common text column names
    for col in df.columns:
        if isinstance(col, str) and col.lower() in common_text_columns:
            text_column = col
            break

    # Step 2: If no common name found, analyze columns to find likely text column
    if not text_column and not selected_column:
        # Get object-type (string) columns
        candidate_columns = [col for col in df.columns if isinstance(col, str) and df[col].dtype == 'object']
        
        if candidate_columns:
            # Score columns based on text likelihood
            column_scores = []
            for col in candidate_columns:
                try:
                    # Calculate average length of non-null, non-empty strings
                    valid_texts = df[col].dropna().astype(str).str.strip()
                    valid_texts = valid_texts[valid_texts != '']
                    if len(valid_texts) == 0:
                        continue
                    avg_length = valid_texts.str.len().mean()
                    # Check if content is mostly non-numeric
                    non_numeric_ratio = valid_texts.apply(
                        lambda x: not re.match(r'^-?\d*\.?\d+$', x)
                    ).mean()
                    # Score: higher for longer text, non-numeric content
                    score = avg_length * non_numeric_ratio
                    column_scores.append((col, score, avg_length))
                except:
                    continue

            # Sort columns by score (descending)
            column_scores = sorted(column_scores, key=lambda x: x[1], reverse=True)

            if column_scores:
                # Auto-select the top-scoring column if it has a reasonable score
                top_column, top_score, avg_length = column_scores[0]
                if top_score > 10:  # Arbitrary threshold (adjust based on testing)
                    text_column = top_column
                    st.write(f"Automatically selected column '{text_column}' (avg. length: {avg_length:.1f} chars).")
                else:
                    # Prompt user to select from top candidates
                    top_candidates = [col for col, _, _ in column_scores[:3]]  # Top 3 candidates
                    selected_column = st.sidebar.selectbox(
                        "Select the column containing text data (recommended based on content):",
                        candidate_columns,
                        index=candidate_columns.index(top_candidates[0]) if top_candidates else 0,
                        key="text_column_select"
                    )
                    st.session_state['selected_column'] = selected_column
                    text_column = selected_column
            else:
                st.error("No columns with valid text data found. Please ensure the file has a text-based column.")
                return None
        else:
            st.error("No valid text columns found in the file. Please ensure the file has a header with a text-based column.")
            return None
    elif selected_column:
        text_column = selected_column

    # Step 3: Validate the selected column
    if not text_column or text_column not in df.columns:
        st.error(f"Selected column '{text_column}' not found in the file. Please ensure the file has a valid text column.")
        return None

    # Validate that the column contains sufficient valid text
    valid_texts = df[text_column].dropna().astype(str).str.strip()
    valid_texts = valid_texts[valid_texts != '']
    if len(valid_texts) < 0.1 * len(df):  # Less than 10% valid text
        st.error(f"Column '{text_column}' contains insufficient valid text data. Please select a different column.")
        return None

    # Rename the identified column to 'Text' for consistency
    if text_column != 'Text':
        df = df.rename(columns={text_column: 'Text'})

    # Drop rows with missing or non-string values in 'Text'
    df = df.dropna(subset=['Text'])
    df = df[df['Text'].apply(lambda x: isinstance(x, str) and x.strip() != '')]

    # If no valid rows remain, return None
    if df.empty:
        st.error("No valid text data found in the 'Text' column after preprocessing.")
        return None

    # Clean text (basic cleaning: remove extra spaces, normalize)
    df['Text'] = df['Text'].apply(lambda x: re.sub(r'\s+', ' ', x.strip()))

    # If 'Sentiment' column exists, remove neutral class (label 1)
    if 'Sentiment' in df.columns:
        df = df[df['Sentiment'] != 1]

    # Reset index
    df = df.reset_index(drop=True)
    return df

def predict_sentiment(texts, batch_size=4):
    """
    Predict sentiment for a list of texts in batches to avoid CUDA OOM.
    Args:
        texts (list or str): List of text strings or a single string.
        batch_size (int): Number of texts to process per batch.
    Returns:
        pd.DataFrame: DataFrame with Text, Sentiment, and Confidence columns.
    """
    if isinstance(texts, str):
        texts = [texts]
    
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encodings = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        try:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                predictions = torch.argmax(probs, dim=1).cpu().numpy()
                confidences = probs.max(dim=1).values.cpu().numpy()
            
            for j, (pred, conf) in enumerate(zip(predictions, confidences)):
                label = "Positive" if pred == 2 else "Negative"
                results.append({"Text": batch_texts[j], "Sentiment": label, "Confidence": conf})
        
        except torch.cuda.OutOfMemoryError:
            st.error("CUDA out of memory. Try reducing batch size or switching to CPU.")
            return None
        
        # Clear memory
        del input_ids, attention_mask, outputs, logits, probs
        torch.cuda.empty_cache()
    
    return pd.DataFrame(results)

# Streamlit interface
st.set_page_config(page_title="Thesis Sentiment Dashboard", layout="wide")
st.title("Sentiment Analysis Dashboard - HieuD's Thesis")
st.write("Binary sentiment analysis based on CardiffNLP RoBERTa - 2025")

# Sidebar: Data source options
st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Select source:", ("Manual Input", "Upload CSV/Excel", "Sample Data from latest.csv"))

# Initialize session state
if "results_df" not in st.session_state:
    st.session_state.results_df = None
if "df_uploaded" not in st.session_state:
    st.session_state.df_uploaded = None
if "selected_column" not in st.session_state:
    st.session_state.selected_column = None
if "columns" not in st.session_state:
    st.session_state.columns = []

# 1. Input Data
st.header("1. Input Data")
input_texts = []
if data_source == "Manual Input":
    user_input = st.text_area("Enter sentences (one per line):", "I love this product\nThis service is terrible")
    input_texts = [line.strip() for line in user_input.split("\n") if line.strip()]
elif data_source == "Upload CSV/Excel":
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file (requires text data)", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                # Try different encodings to handle potential issues
                try:
                    df = pd.read_csv(uploaded_file, header=0, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(uploaded_file, header=0, encoding='latin1')
            else:  # Excel file
                df = pd.read_excel(uploaded_file, header=0, sheet_name=0)
            # Display raw DataFrame for debugging
            st.write("Raw uploaded file (first 5 rows):")
            st.dataframe(df.head())
            df = preprocess_data(df)
            if df is not None:
                input_texts = df["Text"].tolist()
                st.session_state.df_uploaded = df
                st.write("Processed DataFrame (first 5 rows):")
                st.dataframe(df.head())
            else:
                st.stop()
        except Exception as e:
            st.error(f"Error processing file: {str(e)}. Please ensure the file is a valid CSV or Excel with a text column.")
            st.stop()
elif data_source == "Sample Data from latest.csv":
    try:
        df_sample = pd.read_csv(r"C:\Users\hieud\Documents\draft thesis\thesis\src\data\latest.csv").dropna()
        df_sample = df_sample[df_sample["Sentiment"] != 1].sample(100, random_state=42)
        input_texts = df_sample["Text"].tolist()
        st.write("Displaying 5 samples from latest.csv:")
        st.dataframe(df_sample[["Text", "Sentiment"]].head())
    except FileNotFoundError:
        st.error("Sample data file 'latest.csv' not found at the specified path!")
        st.stop()
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

# 3. Model Performance Metrics (unchanged)
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

# 4. Model Comparison (unchanged)
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
    fig_acc = px.line(x=old_model_stats["epochs"], y=[old_model_stats["accuracy"], new_model_stats["accuracy"]],
                      labels={"x": "Epoch", "value": "Accuracy", "variable": "Model"},
                      title="Training Accuracy: Old vs New",
                      color_discrete_map={"wide_variable_0": "orange", "wide_variable_1": "green"})
    fig_acc.data[0].name = "Old Model (Facebook RoBERTa)"
    fig_acc.data[1].name = "New Model (CardiffNLP)"
    st.plotly_chart(fig_acc)

with col2:
    fig_loss = px.line(x=old_model_stats["epochs"], y=[old_model_stats["loss"], new_model_stats["loss"]],
                       labels={"x": "Epoch", "value": "Loss", "variable": "Model"},
                       title="Training Loss: Old vs New",
                       color_discrete_map={"wide_variable_0": "orange", "wide_variable_1": "green"})
    fig_loss.data[0].name = "Old Model (Facebook RoBERTa)"
    fig_loss.data[1].name = "New Model (CardiffNLP)"
    st.plotly_chart(fig_loss)

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

# Footer
st.write("---")
st.write("Thesis project by HieuD - 2025")
