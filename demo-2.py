import gradio as gr
import pandas as pd
import xgboost as xgb
import numpy as np
import shap
import lime
import lime.lime_tabular
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import re
import inspect
import matplotlib.pyplot as plt
import os
import time
import base64
from huggingface_hub import login

# Pre-load models globally
bst = None
gemma_tokenizer = None
gemma_model = None
counter = 0

def show_choice(choice):
    if choice == "Explain Factor":
          counter = 0
    elif choice == "Explain shortly":
          counter = 1
    elif choice == "Explain in detail":
          counter = 2
    return f"You selected: {choice}"

def get_base64_image(image_path):
  with open(image_path, "rb") as img_file:
    return base64.b64encode(img_file.read()).decode('utf-8')


def load_models():
  global bst, gemma_tokenizer, gemma_model
  
  token = input("Enter your Huggingface token for Gemma2 9b:")
  login(token)
  
  # Load the XGBoost model
  bst = xgb.Booster()
  bst.load_model('model/xgb.model')
  
  # Load Gemma model and tokenizer
  quantization_config = BitsAndBytesConfig(load_in_8bit=True)
  gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
  gemma_model = AutoModelForCausalLM.from_pretrained(
      "google/gemma-2-9b-it",
      quantization_config=quantization_config,
  )


def importance_to_str(df, analysis_type):
  if analysis_type == "SHAP":
    value = "SHAP Importance"
  elif analysis_type == "LIME":
    value = "LIME Importance"
  return '\n'.join([f"- ({row['Feature']}, {row[value]})" for _, row in df.iterrows()])


def proba_to_class(proba):
  return "Good" if proba < 0.5 else "Bad"


def format_prompt0(predict_proba_good, predict_proba_bad, predicted_class, shap_analysis, lime_analysis):
  prompt = inspect.cleandoc('''
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
  ''')
  
  return prompt.format(
    predict_proba_good=predict_proba_good,
    predict_proba_bad=predict_proba_bad,
    predicted_class=predicted_class,
    shap_analysis=shap_analysis,
    lime_analysis=lime_analysis
  )

def format_prompt1(predict_proba_good, predict_proba_bad, predicted_class, shap_analysis, lime_analysis):
  prompt = inspect.cleandoc('''
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
  ''')
  
  return prompt.format(
    predict_proba_good=predict_proba_good,
    predict_proba_bad=predict_proba_bad,
    predicted_class=predicted_class,
    shap_analysis=shap_analysis,
    lime_analysis=lime_analysis
  )

def format_prompt0(predict_proba_good, predict_proba_bad, predicted_class, shap_analysis, lime_analysis):
  prompt = inspect.cleandoc('''
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
    - Good: 0.57029474
    - Bad: 0.4297053
    - Predicted to Good

    2. SHAP analysis (Feature, SHAP Importance)
    - (NumSatisfactoryTrades, 0.320243)
    - (PercentTradesNeverDelq, 0.305187)
    - (MSinceMostRecentInqexcl7days, 0.293064)
    - (NumTradesOpeninLast12M, 0.227222)
    - (MaxDelq2PublicRecLast12M, 0.212770)
    - (NumBank2NatlTradesWHighUtilization, 0.073882)
    - (NumTotalTrades, 0.065301)
    - (NumRevolvingTradesWBalance, 0.049232)
    - (MSinceMostRecentDelq, 0.042432)
    - (NumInqLast6M, 0.039852)

    3. LIME analysis (Feature, LIME Importance)
    - (MSinceMostRecentInqexcl7days <= -7.00, -0.186905)
    - (6.00 < MaxDelq2PublicRecLast12M <= 7.00, -0.135623)
    - (96.00 < PercentTradesNeverDelq <= 100.00, -0.124091)
    - (NumSatisfactoryTrades > 27.00, -0.110831)
    - (NumTradesOpeninLast12M > 3.00,  0.058771)
    - (NumRevolvingTradesWBalance > 5.00, 0.036777)
    - (NumTotalTrades > 29.00, -0.027887)
    - (PercentInstallTrades <= 20.00, 0.027751)
    - (NumTrades90Ever2DerogPubRec <= 0.00, -0.019881)
    - (49.50 < PercentTradesWBalance <= 67.00, -0.013193)

    Answer:
    1. SHAP Analysis: First, explain each feature's SHAP importance and how it contributes to the final prediction.
    2. LIME Analysis: Then, explain the individual feature importance from LIME and its role in the prediction.
    3. Conclusion: Finally, synthesize the insights from both SHAP and LIME to provide a comprehensive, easy-to-understand conclusion.
  ''')
  
  return prompt.format(
    predict_proba_good=predict_proba_good,
    predict_proba_bad=predict_proba_bad,
    predicted_class=predicted_class,
    shap_analysis=shap_analysis,
    lime_analysis=lime_analysis
  )

def generate_gemma_response(prompt, tokenizer, model, device):
  inputs = tokenizer(prompt, return_tensors="pt").to(device)
  output = model.generate(inputs["input_ids"], max_length=2500, num_return_sequences=1)
  response = tokenizer.decode(output[0], skip_special_tokens=True)
  return response


def extract_conclusion_part(gemma_response):
  if "Conclusion:" in gemma_response:
    return gemma_response.split("Conclusion:")[1].strip()
  else:
    conclusions = re.findall(r'<Conclusion>(.*?)</Conclusion>', gemma_response, re.DOTALL)
    return conclusions[-1].strip()


def gemma_analysis(prompt):
  global gemma_model, gemma_tokenizer 

  device = torch.device("cuda:0")
  
  response = generate_gemma_response(prompt, gemma_tokenizer, gemma_model, device)
  conclusion = extract_conclusion_part(response)
  
  return conclusion


def get_input(user_input_values):
  if all(value == 0 for value in user_input_values):
    df = pd.read_csv('dataset/heloc.csv')
    df.drop(columns=['RiskPerformance'], inplace=True)
    user_data = df.sample(1)  # Load one sample
  else:
    user_data = pd.DataFrame({
      'ExternalRiskEstimate': [user_input_values[0]],
      'MSinceOldestTradeOpen': [user_input_values[1]],
      'MSinceMostRecentTradeOpen': [user_input_values[2]],
      'AverageMInFile': [user_input_values[3]],
      'NumSatisfactoryTrades': [user_input_values[4]],
      'NumTrades60Ever2DerogPubRec': [user_input_values[5]],
      'NumTrades90Ever2DerogPubRec': [user_input_values[6]],
      'PercentTradesNeverDelq': [user_input_values[7]],
      'MSinceMostRecentDelq': [user_input_values[8]],
      'MaxDelq2PublicRecLast12M': [user_input_values[9]],
      'MaxDelqEver': [user_input_values[10]],
      'NumTotalTrades': [user_input_values[11]],
      'NumTradesOpeninLast12M': [user_input_values[12]],
      'PercentInstallTrades': [user_input_values[13]],
      'MSinceMostRecentInqexcl7days': [user_input_values[14]],
      'NumInqLast6M': [user_input_values[15]],
      'NumInqLast6Mexcl7days': [user_input_values[16]],
      'NetFractionRevolvingBurden': [user_input_values[17]],
      'NetFractionInstallBurden': [user_input_values[18]],
      'NumRevolvingTradesWBalance': [user_input_values[19]],
      'NumInstallTradesWBalance': [user_input_values[20]],
      'NumBank2NatlTradesWHighUtilization': [user_input_values[21]],
      'PercentTradesWBalance': [user_input_values[22]],
    })
      
  return user_data


def analysis(
    ExternalRiskEstimate, MSinceOldestTradeOpen, MSinceMostRecentTradeOpen, 
    AverageMInFile, NumSatisfactoryTrades, NumTrades60Ever2DerogPubRec, 
    NumTrades90Ever2DerogPubRec, PercentTradesNeverDelq, MSinceMostRecentDelq, 
    MaxDelq2PublicRecLast12M, MaxDelqEver, NumTotalTrades, NumTradesOpeninLast12M, 
    PercentInstallTrades, MSinceMostRecentInqexcl7days, NumInqLast6M, 
    NumInqLast6Mexcl7days, NetFractionRevolvingBurden, NetFractionInstallBurden, 
    NumRevolvingTradesWBalance, NumInstallTradesWBalance, 
    NumBank2NatlTradesWHighUtilization, PercentTradesWBalance
):
  global bst, gemma_tokenizer, gemma_model

  # List of all user inputs
  user_input_values = [
    ExternalRiskEstimate, MSinceOldestTradeOpen, MSinceMostRecentTradeOpen, 
    AverageMInFile, NumSatisfactoryTrades, NumTrades60Ever2DerogPubRec, 
    NumTrades90Ever2DerogPubRec, PercentTradesNeverDelq, MSinceMostRecentDelq, 
    MaxDelq2PublicRecLast12M, MaxDelqEver, NumTotalTrades, NumTradesOpeninLast12M, 
    PercentInstallTrades, MSinceMostRecentInqexcl7days, NumInqLast6M, 
    NumInqLast6Mexcl7days, NetFractionRevolvingBurden, NetFractionInstallBurden, 
    NumRevolvingTradesWBalance, NumInstallTradesWBalance, 
    NumBank2NatlTradesWHighUtilization, PercentTradesWBalance
  ]

  # Construct user data
  
  user_data = get_input(user_input_values)
  
  # Perform prediction
  d_user_data = xgb.DMatrix(user_data)
  user_pred = bst.predict(d_user_data)
  prediction_proba = user_pred[0]
  prediction_class = proba_to_class(user_pred[0])
  
  # SHAP and LIME Analysis (similar to before)
  explainer = shap.TreeExplainer(bst)
  shap_values = explainer.shap_values(user_data)

  lime_explainer = lime.lime_tabular.LimeTabularExplainer(
      user_data.values,
      feature_names=list(user_data.columns),
      class_names=['Good', 'Bad'],
      discretize_continuous=True
  )
  # XGBoost의 예측을 확률로 반환하도록 하는 래핑된 함수
  def predict_fn_for_lime(input_data):
      dmatrix_data = xgb.DMatrix(input_data, feature_names=list(user_data.columns))
      # 확률 값으로 반환하여 LIME에서 사용 가능하게 함
      probas = bst.predict(dmatrix_data)  # 확률 값 반환
      return np.column_stack([1 - probas, probas])  # LIME은 클래스별 확률을 받으므로 [1-p, p] 형식으로 반환

  lime_exp = lime_explainer.explain_instance(user_data.iloc[0].values, predict_fn_for_lime, num_features=10)
  lime_importance = pd.DataFrame(lime_exp.as_list(), columns=['Feature', 'LIME Importance'])
  lime_importance_string = importance_to_str(lime_importance, "LIME")

  # Generate LIME plot
  now = time.time()
  os.makedirs(f"images/{now}", exist_ok=True)
  lime_plot = lime_exp.as_pyplot_figure()  # Get the LIME plot as a figure
  lime_plot.savefig(f"images/{now}/lime_plot.png", bbox_inches="tight")  # Save plot to file
  plt.close(lime_plot)

  shap_importance = pd.DataFrame({'Feature': user_data.columns, 'SHAP Importance': np.abs(shap_values[0])})
  shap_importance = shap_importance.sort_values(by='SHAP Importance', ascending=False).head(10)
  shap_importance_string = importance_to_str(shap_importance, "SHAP")

  # Generate SHAP plot
  plt.figure()
  shap.summary_plot(shap_values, user_data, show=False)  # Generate SHAP summary plot
  plt.savefig(f"images/{now}/shap_summary_plot.png", bbox_inches="tight")  # Save plot to file
  plt.close()

  # Generate prompt for Gemma analysis
  good_proba = 1 - prediction_proba
  bad_proba = prediction_proba
  if counter == 0:
    prompt = format_prompt0(good_proba, bad_proba, prediction_class, shap_importance_string, lime_importance_string)
  if counter == 1:
    prompt = format_prompt1(good_proba, bad_proba, prediction_class, shap_importance_string, lime_importance_string)
  if counter == 2:
    prompt = format_prompt2(good_proba, bad_proba, prediction_class, shap_importance_string, lime_importance_string)
  
  # Use Gemma for final analysis
  conclusion = gemma_analysis(prompt)
  
  return conclusion, f"images/{now}/shap_summary_plot.png", f"images/{now}/lime_plot.png"


load_models()
image_base64 = get_base64_image("asset/logo.png")

# Use Gradio Blocks for advanced layout
with gr.Blocks() as demo:
  with gr.Row():
    gr.HTML(f"""
      <div style="display: flex; align-items: center; justify-content: flex-start;">
        <img src="data:image/png;base64,{image_base64}" alt="Logo" style="width: 80px; height: auto; margin-right: 10px;">
        <h1 style="margin: 0; padding: 0;">MMExplainer: Credit Risk Prediction</h1>
      </div>
    """)  # Align logo and title on the same row, with left justification
  gr.Markdown("MMExplainer analyzes ML model results using HELOC data with XAI methods like LIME and SHAP, providing clear, Gemma-based insights. It enhances transparency and trust in predictions, helping users make informed decisions in finance and credit risk management.")
  
  inputs = []
  with gr.Row():
    inputs.append(gr.Number(label="ExternalRiskEstimate", info="외부 신용점수 추정치"))
    inputs.append(gr.Number(label="MSinceOldestTradeOpen", info="가장 오래된 거래가 열린 이후 경과한 개월 수"))
    inputs.append(gr.Number(label="MSinceMostRecentTradeOpen", info="최근 거래가 열린 이후 경과한 개월 수"))
    inputs.append(gr.Number(label="AverageMInFile", info="신용 기록 파일 내 평균 월수"))
  with gr.Row():
    inputs.append(gr.Number(label="NumSatisfactoryTrades", info="만족스러운 거래 수"))
    inputs.append(gr.Number(label="NumTrades60Ever2DerogPubRec", info="60일 이상 연체 또는 부정적 공공 기록이 있는 거래 수"))
    inputs.append(gr.Number(label="NumTrades90Ever2DerogPubRec", info="90일 이상 연체된 거래 수"))
    inputs.append(gr.Number(label="PercentTradesNeverDelq", info="연체되지 않은 거래의 비율"))
  with gr.Row():
    inputs.append(gr.Number(label="MSinceMostRecentDelq", info="최근 연체 이후 경과한 개월 수"))
    inputs.append(gr.Number(label="MaxDelq2PublicRecLast12M", info="지난 12개월 동안의 공공 기록에서 가장 큰 연체 기록"))
    inputs.append(gr.Number(label="MaxDelqEver", info="최대 연체 기록"))
    inputs.append(gr.Number(label="NumTotalTrades", info="전체 거래 수"))
  with gr.Row():
    inputs.append(gr.Number(label="NumTradesOpeninLast12M", info="최근 12개월 동안 개설된 거래 수"))
    inputs.append(gr.Number(label="PercentInstallTrades", info="할부 거래 비율"))
    inputs.append(gr.Number(label="MSinceMostRecentInqexcl7days", info="최근 7일 제외, 신용 조회 이후 경과한 개월 수"))
    inputs.append(gr.Number(label="NumInqLast6M", info="지난 6개월 동안 신용 조회 수"))
  with gr.Row():
    inputs.append(gr.Number(label="NumInqLast6Mexcl7days", info="지난 6개월 동안 신용 조회 수, 최근 7일 제외"))
    inputs.append(gr.Number(label="NetFractionRevolvingBurden", info="순회전 부채 부담 비율"))
    inputs.append(gr.Number(label="NetFractionInstallBurden", info="할부 부채 부담 비율"))
    inputs.append(gr.Number(label="NumRevolvingTradesWBalance", info="잔액이 있는 회전 신용 거래 수"))
  with gr.Row():
    inputs.append(gr.Number(label="NumInstallTradesWBalance", info="잔액이 있는 할부 신용 거래 수"))
    inputs.append(gr.Number(label="NumBank2NatlTradesWHighUtilization", info="높은 사용 비율을 보이는 거래 수"))
    inputs.append(gr.Number(label="PercentTradesWBalance", info="잔액이 있는 거래 비율"))

  with gr.Row():
    radio = gr.Radio(["Explain Factor", "Explain shortly", "Explain in detail"], label="Choose an option")
    submit_btn = gr.Button("Submit")
  output_text = gr.Textbox(label="Gemma Analysis Conclusion")
  with gr.Row():
    shap_image = gr.Image(label="SHAP Summary Plot")  # For SHAP plot
    lime_image = gr.Image(label="LIME Explanation Plot")  # For LIME plot

  submit_btn.click(
    fn=analysis, 
    inputs=inputs, 
    outputs=[output_text, shap_image, lime_image],
  )


# Launch the Gradio app
demo.launch(share=True)