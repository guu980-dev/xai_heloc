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
from dotenv import load_dotenv
from prompt import friendly_tone_template, formal_tone_template

# Pre-load models globally
bst = None
gemma_tokenizer = None
gemma_model = None


def get_base64_image(image_path):
  with open(image_path, "rb") as img_file:
    return base64.b64encode(img_file.read()).decode('utf-8')


def get_base64_image(image_path):
  with open(image_path, "rb") as img_file:
    return base64.b64encode(img_file.read()).decode('utf-8')


def load_models():
  global bst, gemma_tokenizer, gemma_model

  load_dotenv()
  login(os.getenv("HUGGINGFACE_TOKEN"))
  
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


def format_prompt(predict_proba_good, predict_proba_bad, predicted_class, shap_analysis, lime_analysis, prompt_type):
  if prompt_type == "Friendly Tone":
    prompt_templtae = friendly_tone_template
  elif prompt_type == "Formal Tone":
    prompt_templtae = formal_tone_template
    
  prompt_template = inspect.cleandoc(prompt_templtae)
  return prompt_template.format(
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


# User Only Prompt 2.x versions
def extract_conclusion_part(gemma_response):
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
    NumBank2NatlTradesWHighUtilization, PercentTradesWBalance,
    prompt_type
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
  
  prompt = format_prompt(good_proba, bad_proba, prediction_class, shap_importance_string, lime_importance_string, prompt_type)

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
    radio = gr.Radio(["Friendly Tone", "Formal Tone"], label="Choose Prompt Option")
    inputs.append(radio)
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
