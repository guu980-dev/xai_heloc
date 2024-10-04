<div align="center">
    <img src="./asset/logo.png" alt="MMExplainer Logo" width="300"/>
</div>

# MMExplainer

MMExplainer is an AI-driven solution designed to predict RiskPerformance on the HELOC (Home Equity Line of Credit) dataset using an XGBoost model. In addition to delivering accurate predictions, the system provides detailed insights into the reasons behind each prediction through Explainable AI (XAI) techniques like SHAP and LIME. The results are further refined using the Gemma2-9b model to offer a comprehensive explanation. A Gradio-powered demo is included to showcase these capabilities.

## Features

- **XGBoost Model**: Trains on the HELOC dataset to predict RiskPerformance.
- **Explainability**: Utilizes SHAP and LIME for feature-level explanations of model predictions.
- **Advanced Insights**: Employs the Gemma2-9b model to generate in-depth explanations.
- **Zero-shot CoT Prompts**: Uses Chain-of-Thought reasoning for enhanced zero-shot prompting.
- **Gradio Demo**: Provides an interactive demo to visualize model predictions and explanations.

## Getting Started

### Setup Instructions

1. Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file in the root directory and add your Hugging Face API token:
   ```
   HUGGINGFACE_TOKEN=your_huggingface_token
   ```
3. Prepare the HELOC dataset and place the `heloc.csv` file in the `./dataset/` folder.
4. Train the XGBoost model by executing the `train_xgb.ipynb` notebook.
5. Launch the Gradio demo by running:
   ```bash
   python demo.py
   ```

## Future Work

- Integrate additional machine learning models.
- Test on other financial datasets.
- Explore more advanced XAI techniques for better interpretability.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
