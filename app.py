"""
Streamlit Dashboard for Customer Churn Prediction.

This interactive dashboard allows users to:
- Upload customer data and get churn predictions
- Manually input customer features
- View prediction explanations (SHAP)
- Explore model performance metrics
"""

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

from src.explainability import ModelExplainer
from src.utils import get_config


# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model_and_preprocessor(model_path: str, preprocessor_path: str):
    """Load trained model and preprocessor."""
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model or preprocessor: {e}")
        return None, None


@st.cache_data
def load_evaluation_results(results_path: str):
    """Load evaluation results."""
    try:
        import json
        with open(results_path, "r") as f:
            results = json.load(f)
        return results
    except Exception as e:
        st.warning(f"Could not load evaluation results: {e}")
        return None


def create_input_form(feature_names: list):
    """Create input form for manual feature entry."""
    st.subheader("📝 Manual Feature Input")

    # This is a simplified version - in reality, you'd customize based on your features
    # For the Telco dataset, here are common features:

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    with col2:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with col3:
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    col4, col5 = st.columns(2)

    with col4:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])

    with col5:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, 0.1)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0, 1.0)

    # Create DataFrame
    input_data = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    return pd.DataFrame([input_data])


def preprocess_input(df: pd.DataFrame, preprocessor_state):
    """Preprocess input data using the trained preprocessor."""
    try:
        from src.preprocessing import DataPreprocessor

        # Recreate preprocessor from saved state
        preprocessor = DataPreprocessor()
        preprocessor.label_encoders = preprocessor_state['label_encoders']
        preprocessor.scaler = preprocessor_state['scaler']
        preprocessor.feature_names = preprocessor_state['feature_names']
        preprocessor.target_encoder = preprocessor_state['target_encoder']
        preprocessor.config = preprocessor_state['config']

        # Clean data (no columns to drop for inference)
        df_clean = df.copy()

        # Convert TotalCharges to numeric if needed
        if 'TotalCharges' in df_clean.columns:
            df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

        # Encode categorical
        df_encoded = preprocessor.encode_categorical(df_clean, fit=False)

        # Ensure all expected features are present
        for feature in preprocessor.feature_names:
            if feature not in df_encoded.columns:
                df_encoded[feature] = 0

        # Select only the features used during training
        df_encoded = df_encoded[preprocessor.feature_names]

        # Scale features
        df_scaled = preprocessor.scale_features(df_encoded, fit=False)

        return df_scaled

    except Exception as e:
        st.error(f"Error preprocessing input: {e}")
        return None


def predict_churn(model, X, preprocessor_state):
    """Make churn prediction."""
    try:
        # Prediction
        prediction = model.predict(X)[0]
        prediction_proba = model.predict_proba(X)[0]

        # Decode prediction
        if 'target_encoder' in preprocessor_state and preprocessor_state['target_encoder']:
            prediction_label = preprocessor_state['target_encoder'].inverse_transform([prediction])[0]
        else:
            prediction_label = "Yes" if prediction == 1 else "No"

        churn_probability = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]

        return prediction_label, churn_probability

    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None


def display_prediction(prediction_label, churn_probability):
    """Display prediction results."""
    st.subheader("🎯 Prediction Result")

    # Create columns for display
    col1, col2 = st.columns(2)

    with col1:
        # Churn status
        if prediction_label == "Yes":
            st.error(f"### ⚠️ Churn Predicted: **{prediction_label}**")
        else:
            st.success(f"### ✅ Churn Predicted: **{prediction_label}**")

    with col2:
        # Probability gauge
        st.metric("Churn Probability", f"{churn_probability:.1%}")

    # Probability bar
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_probability * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Churn Risk Level"},
        gauge={
            "axis": {"range": [None, 100]},
            "bar": {"color": "red" if churn_probability > 0.5 else "green"},
            "steps": [
                {"range": [0, 33], "color": "lightgreen"},
                {"range": [33, 66], "color": "yellow"},
                {"range": [66, 100], "color": "lightcoral"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 4},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))

    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def display_shap_explanation(model, X, preprocessor_state):
    """Display SHAP explanation for the prediction."""
    st.subheader("🔍 Prediction Explanation (SHAP)")

    with st.spinner("Generating explanation..."):
        try:
            import shap

            # For single prediction, we need background data
            # Load training data if available, otherwise use the input as background
            try:
                # Try to load preprocessed training data
                X_train = pd.read_csv("data/processed/X_train.csv")
                if len(X_train) > 100:
                    background_data = X_train.sample(100, random_state=42)
                else:
                    background_data = X_train
            except FileNotFoundError:
                # Fallback: use the input data repeated as background
                st.info("Using limited background data. For better explanations, ensure training data is available.")
                background_data = pd.concat([X] * 50, ignore_index=True)

            # Create appropriate SHAP explainer based on model type
            model_name = str(type(model).__name__).lower()

            if 'logistic' in model_name or 'linear' in model_name:
                # For linear models, use LinearExplainer or masker-based Explainer
                try:
                    # Try using LinearExplainer for logistic regression
                    explainer = shap.LinearExplainer(model, background_data)
                    shap_values = explainer.shap_values(X)
                    st.caption("Using SHAP LinearExplainer (fast, exact)")
                except Exception as e:
                    # Fallback to general Explainer with masker
                    masker = shap.maskers.Independent(background_data, max_samples=100)
                    explainer = shap.Explainer(model.predict_proba, masker)
                    shap_values_obj = explainer(X)
                    # Extract values for positive class (churn)
                    shap_values = shap_values_obj.values[:, :, 1] if len(shap_values_obj.values.shape) == 3 else shap_values_obj.values
                    st.caption("Using SHAP Explainer with masker")
            elif 'tree' in model_name or 'forest' in model_name or 'xgb' in model_name or 'gradient' in model_name:
                # For tree-based models
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                # For binary classification, get positive class
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                st.caption("Using SHAP TreeExplainer (fast, exact)")
            else:
                # Generic explainer for other models
                explainer = shap.KernelExplainer(model.predict_proba, background_data)
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                st.caption("Using SHAP KernelExplainer (slower, model-agnostic)")

            # Extract SHAP values for the first instance
            # Handle different SHAP value shapes
            if len(shap_values.shape) == 3:
                # Shape: [n_samples, n_features, n_classes]
                # For binary classification, use positive class (index 1)
                instance_shap = shap_values[0, :, 1]
            elif len(shap_values.shape) == 2:
                # Shape: [n_samples, n_features]
                instance_shap = shap_values[0]
            elif len(shap_values.shape) == 1:
                # Shape: [n_features]
                instance_shap = shap_values
            else:
                raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")

            # Ensure it's 1D
            if len(instance_shap.shape) > 1:
                instance_shap = instance_shap.flatten()

            # Create importance dataframe
            importance = pd.DataFrame({
                "feature": preprocessor_state['feature_names'][:len(instance_shap)],
                "shap_value": instance_shap.flatten() if hasattr(instance_shap, 'flatten') else instance_shap,
                "feature_value": X.iloc[0].values[:len(instance_shap)].flatten() if hasattr(X.iloc[0].values[:len(instance_shap)], 'flatten') else X.iloc[0].values[:len(instance_shap)],
            })

            importance["abs_shap"] = importance["shap_value"].abs()
            importance = importance.sort_values("abs_shap", ascending=False).head(10)

            # Check if we have meaningful SHAP values
            if importance["abs_shap"].sum() < 0.001:
                st.warning("SHAP values are very close to zero. This might indicate an issue with the explanation.")
                st.write("Debug info:", importance)

            # Plot - Create a proper color scale for positive/negative values
            fig = go.Figure(go.Bar(
                x=importance['shap_value'],
                y=importance['feature'],
                orientation='h',
                marker=dict(
                    color=importance['shap_value'],
                    colorscale=[[0, 'green'], [0.5, 'lightgray'], [1, 'red']],
                    cmid=0,
                    showscale=True,
                    colorbar=dict(title="SHAP Value")
                ),
                text=importance['shap_value'].round(3),
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:.3f}<extra></extra>'
            ))

            fig.update_layout(
                title={
                    'text': "Top 10 Features Influencing This Prediction",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': 'white'}
                },
                xaxis_title="SHAP Value (Impact on Prediction)",
                yaxis_title="Feature",
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                showlegend=False,
                yaxis={'categoryorder': 'total ascending'}
            )

            st.plotly_chart(fig, use_container_width=True)

            # Explanation text
            st.info(
                """
                **How to interpret:**
                - **Red bars (positive values)**: Features pushing prediction towards **Churn**
                - **Green bars (negative values)**: Features pushing prediction towards **No Churn**
                - **Longer bars**: Stronger influence on the prediction
                """
            )

        except Exception as e:
            st.error(f"Could not generate SHAP explanation: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.info("SHAP explanations require the model to be compatible with SHAP explainers.")


def main():
    """Main Streamlit app."""
    st.title("📊 Customer Churn Prediction System")
    st.markdown("### Predict customer churn with explainable AI")

    # Sidebar
    st.sidebar.title("⚙️ Configuration")

    # Model selection
    models_dir = Path("data/models")
    if not models_dir.exists():
        st.error("Models directory not found. Please run the training pipeline first.")
        st.stop()

    available_models = list(models_dir.glob("*.joblib"))
    available_models = [m for m in available_models if m.stem != "preprocessor"]

    if not available_models:
        st.error("No trained models found. Please run the training pipeline first.")
        st.stop()

    model_names = [m.stem for m in available_models]
    selected_model = st.sidebar.selectbox("Select Model", model_names)

    # Load model and preprocessor
    model_path = models_dir / f"{selected_model}.joblib"
    preprocessor_path = models_dir / "preprocessor.joblib"

    model, preprocessor_state = load_model_and_preprocessor(str(model_path), str(preprocessor_path))

    if model is None or preprocessor_state is None:
        st.error("Failed to load model or preprocessor.")
        st.stop()

    st.sidebar.success(f"✅ Loaded model: **{selected_model}**")

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["📝 Single Prediction", "📂 Batch Prediction", "📈 Model Performance"])

    # Tab 1: Single Prediction
    with tab1:
        st.header("Single Customer Prediction")

        input_df = create_input_form(preprocessor_state['feature_names'])

        if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
            # Preprocess
            X_processed = preprocess_input(input_df, preprocessor_state)

            if X_processed is not None:
                # Predict
                prediction_label, churn_probability = predict_churn(
                    model, X_processed, preprocessor_state
                )

                if prediction_label is not None:
                    # Display results
                    display_prediction(prediction_label, churn_probability)

                    # SHAP explanation
                    display_shap_explanation(model, X_processed, preprocessor_state)

    # Tab 2: Batch Prediction
    with tab2:
        st.header("Batch Prediction")
        st.markdown("Upload a CSV file with customer data to get predictions for multiple customers.")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            # Read file
            df_upload = pd.read_csv(uploaded_file)

            st.subheader("📋 Uploaded Data Preview")
            st.dataframe(df_upload.head())

            if st.button("🔮 Predict All", type="primary"):
                with st.spinner("Processing predictions..."):
                    # Preprocess
                    X_processed = preprocess_input(df_upload, preprocessor_state)

                    if X_processed is not None:
                        # Predict
                        predictions = model.predict(X_processed)
                        probabilities = model.predict_proba(X_processed)[:, 1]

                        # Add to dataframe
                        df_results = df_upload.copy()
                        df_results["Predicted_Churn"] = preprocessor_state['target_encoder'].inverse_transform(predictions)
                        df_results["Churn_Probability"] = probabilities

                        st.subheader("📊 Prediction Results")
                        st.dataframe(df_results)

                        # Summary
                        churn_count = (df_results["Predicted_Churn"] == "Yes").sum()
                        churn_rate = churn_count / len(df_results) * 100

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Customers", len(df_results))
                        col2.metric("Predicted Churners", churn_count)
                        col3.metric("Predicted Churn Rate", f"{churn_rate:.1f}%")

                        # Download button
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="churn_predictions.csv",
                            mime="text/csv",
                        )

    # Tab 3: Model Performance
    with tab3:
        st.header("Model Performance Metrics")

        # Load evaluation results
        results_path = Path("data/results/evaluation_results.json")
        results = load_evaluation_results(str(results_path))

        if results:
            # Model comparison
            st.subheader("📊 Model Comparison")

            comparison_data = []
            for model_name, metrics in results.items():
                comparison_data.append({
                    "Model": model_name,
                    "Accuracy": metrics.get("accuracy", 0),
                    "Precision": metrics.get("precision", 0),
                    "Recall": metrics.get("recall", 0),
                    "F1 Score": metrics.get("f1", 0),
                    "ROC-AUC": metrics.get("roc_auc", 0),
                })

            comparison_df = pd.DataFrame(comparison_data)

            # Display table
            st.dataframe(comparison_df, use_container_width=True)

            # Metrics chart
            fig = px.bar(
                comparison_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
                x="Metric",
                y="Score",
                color="Model",
                barmode="group",
                title="Model Performance Comparison",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Confusion matrix for selected model
            if selected_model in results:
                st.subheader(f"📋 Confusion Matrix - {selected_model}")

                metrics = results[selected_model]
                cm_data = pd.DataFrame(
                    [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
                    columns=["Predicted No", "Predicted Yes"],
                    index=["Actual No", "Actual Yes"],
                )

                fig = px.imshow(
                    cm_data,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    title=f"Confusion Matrix - {selected_model}",
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No evaluation results found. Run the training pipeline to generate metrics.")

        # Display plots if available
        plots_dir = Path("data/plots")
        if plots_dir.exists():
            st.subheader("📈 Visualizations")

            plot_files = {
                "ROC Curves": "roc_curves.png",
                "Model Comparison": "model_comparison.png",
                "Churn Distribution": "churn_distribution.png",
                "SHAP Summary": "shap_summary.png",
            }

            for title, filename in plot_files.items():
                plot_path = plots_dir / filename
                if plot_path.exists():
                    with st.expander(f"📊 {title}"):
                        st.image(str(plot_path), use_container_width=True)


if __name__ == "__main__":
    main()
