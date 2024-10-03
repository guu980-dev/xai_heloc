# Ver 2.5: Unexpert user friendly
friendly_tone_template = '''
  Question:
  The following is the result of binary classification using the HELOC (Home Equity Line of Credit) Dataset and XGBClassifier to classify RiskPerformance into “Good” and “Bad.”
  The value “Bad” indicates that a consumer was 90 days past due or worse at least once over a period of 24 months from when the credit account was opened. The value “Good” indicates that they have made their payments without ever being more than 90 days overdue.

  Before answering, please think steyp by step coincisely in these steps to explain the prediction.
  1. SHAP Analysis: Analyze the key features from the SHAP analysis, explaining how each feature contributes to the prediction.
  This step should be inside <SHAP>$$INSERT TEXT HERE$$</SHAP> tag.
  2. LIME Analysis: Analyze the key features from the LIME analysis, explaining the contribution of each feature in terms of how it influences the prediction.
  This step should be inside <LIME>$$INSERT TEXT HERE$$</LIME> tag.
  3. Insight Synthesis: Based on the individual feature analyses from SHAP and LIME, synthesize the insights to provide a comprehensive conclusion. The conclusion should focus on how these features work together to influence the final prediction.
  This step should be inside <Insight>$$INSERT TEXT HERE$$</Insight> tag.
  4. Final Explanation for Non-Experts: Provide the prediction result and explain the comprehensive reasoning behind the result, considering multiple factors that contributed to this outcome. Ensure the explanation is clear, detailed, and avoids using technical terms or direct references to probabilities or numbers, so that the final explanation is understandable to non-experts in machine learning or finance.
  This step should be inside <Conclusion>$$INSERT TEXT HERE$$</Conclusion> tag.
  In this part,
  - Ensure to be thorough and specific as possible, with enough length to fully explain the reasoning behind the prediction and offer clear, actionable advice to the user.
  - Please respond as if you were a human, using natural conversational tone. Be engaging, empathetic, and use phrases and expressions that sound like they’re coming from a real person, keeping the tone friendly and conversational. Avoid sounding overly formal or robotic.
  - Please provide a sentences without explicitly using terms like 'model,' 'probability,' or directly mentioning numbers. Instead, explain the concepts in simple, intuitive language that avoids technical jargon.
  - At the end of the part, provide a personalized piece of advice for the user on how they can improve or maintain their risk performance in the future.

  Context:
  1. Prediction Probability
  - Good: {predict_proba_good}
  - Bad: {predict_proba_bad}
  - Predicted to {predicted_class}

  2. SHAP analysis (Feature, SHAP Importance)
  {shap_analysis}

  3. LIME analysis (Feature, LIME Importance)
  {lime_analysis}

  Answer:
'''

formal_tone_template = '''
  Question:
  The following is the result of binary classification using the HELOC (Home Equity Line of Credit) Dataset and XGBClassifier to classify RiskPerformance into “Good” and “Bad.”
  The value “Bad” indicates that a consumer was 90 days past due or worse at least once over a period of 24 months from when the credit account was opened. The value “Good” indicates that they have made their payments without ever being more than 90 days overdue.

  Before answering, please think steyp by step coincisely in these steps to explain the prediction.
  1. SHAP Analysis: Analyze the key features from the SHAP analysis, explaining how each feature contributes to the prediction.
  This step should be inside <SHAP>$$INSERT TEXT HERE$$</SHAP> tag.
  2. LIME Analysis: Analyze the key features from the LIME analysis, explaining the contribution of each feature in terms of how it influences the prediction.
  This step should be inside <LIME>$$INSERT TEXT HERE$$</LIME> tag.
  3. Insight Synthesis: Based on the individual feature analyses from SHAP and LIME, synthesize the insights to provide a comprehensive conclusion. The conclusion should focus on how these features work together to influence the final prediction.
  This step should be inside <Insight>$$INSERT TEXT HERE$$</Insight> tag.
  4. Final Explanation for Non-Experts: Provide the prediction result and explain the comprehensive reasoning behind the result, considering multiple factors that contributed to this outcome. Ensure the explanation is clear, detailed, and avoids using technical terms or direct references to probabilities or numbers, so that the final explanation is understandable to non-experts in machine learning or finance.
  This step should be inside <Conclusion>$$INSERT TEXT HERE$$</Conclusion> tag.

  Context:
  1. Prediction Probability
  - Good: {predict_proba_good}
  - Bad: {predict_proba_bad}
  - Predicted to {predicted_class}

  2. SHAP analysis (Feature, SHAP Importance)
  {shap_analysis}

  3. LIME analysis (Feature, LIME Importance)
  {lime_analysis}

  Answer:
'''

# Ver 1.5: More anayltic
analytic_tone_template = '''
  Question:
  The following is the result of binary classification using the HELOC (Home Equity Line of Credit) Dataset and XGBClassifier to classify RiskPerformance into “Good” and “Bad.”
  The value “Bad” indicates that a consumer was 90 days past due or worse at least once over a period of 24 months from when the credit account was opened. The value “Good” indicates that they have made their payments without ever being more than 90 days overdue.

  Please follow these steps to explain the prediction:

  1. Analyze the key features from the SHAP analysis, explaining how each feature contributes to the prediction.
  2. Analyze the key features from the LIME analysis, explaining the contribution of each feature in terms of how it influences the prediction.
  3. Based on the individual feature analyses from SHAP and LIME, synthesize the insights to provide a comprehensive conclusion. The conclusion should focus on how these features work together to influence the final prediction.
  4. Instead of focusing on technical jargon, ensure that the final explanation is understandable to non-experts in machine learning or finance.

  Context:
  1. Prediction Probability
  - Good: {predict_proba_good}
  - Bad: {predic_proba_bad}
  - Predicted to {predicted_class}

  2. SHAP analysis (Feature, SHAP Importance)
  {shap_analysis}

  3. LIME analysis (Feature, LIME Importance)
  {lime_analysis}

  Answer:
  1. SHAP Analysis: First, explain each feature's SHAP importance and how it contributes to the final prediction.
  2. LIME Analysis: Then, explain the individual feature importance from LIME and its role in the prediction.
  3. Conclusion: Finally, synthesize the insights from both SHAP and LIME to provide a comprehensive, easy-to-understand conclusion.
'''