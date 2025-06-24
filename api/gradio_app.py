import gradio as gr
import requests
import json

FASTAPI_PREDICT_URL = "http://127.0.0.1:8000/predict"
LABEL_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
GOOD_COMMENT_THRESHOLD = 0.3

def get_toxicity_analysis(comment_text):
    payload = {"text": comment_text}
    overall_status_text = "Status: Could not connect or get a valid response from the API."
    details_html = "<p><em>No details available due to an error.</em></p>"

    try:
        response = requests.post(FASTAPI_PREDICT_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        
        label_probabilities = result.get("label_probabilities", {})
        is_good_comment = True
        detected_toxicities_list = []

        for label in LABEL_COLUMNS:
            prob = label_probabilities.get(label, 0.0)
            if prob >= GOOD_COMMENT_THRESHOLD:
                is_good_comment = False
            if prob >= GOOD_COMMENT_THRESHOLD: 
                 detected_toxicities_list.append({
                     "name": label.replace("_", " ").capitalize(), 
                     "score": prob 
                 })
        
        if is_good_comment:
            overall_status_text = "Status: This appears to be a Good Comment."
            details_html = "<p style='color: green; font-weight: bold;'>No specific toxicities detected above the threshold.</p>"
        else:
            overall_status_text = "Status: Potential Toxicity Detected."
            if detected_toxicities_list:
                details_html = "<div style='font-family: sans-serif;'>"
                details_html += "<h4 style='margin-bottom: 10px;'>Detected Categories:</h4>"
                
                for item in detected_toxicities_list:
                    score_percentage = item['score'] * 100
                    score_display = f"{item['score']:.2f}"
                    
                    bar_color = "green"
                    if item['score'] >= 0.7:
                        bar_color = "red"
                    elif item['score'] >= 0.4:
                        bar_color = "orange"
                    else: 
                        bar_color = "gold" 

                    details_html += f"""
                    <div style="margin-bottom: 8px;">
                        <div style="display: flex; align-items: center; margin-bottom: 2px;">
                            <span style="font-weight: bold; width: 120px; display: inline-block;">{item['name']}:</span>
                            <span style="font-size: 0.9em; color: #333;">{score_display}</span>
                        </div>
                        <div style="width: 100%; background-color: #e9ecef; border-radius: 4px; overflow: hidden;">
                            <div style="width: {score_percentage}%; background-color: {bar_color}; height: 18px; 
                                        line-height: 18px; color: white; text-align: right; padding-right: 5px;
                                        font-size: 0.8em; transition: width 0.3s ease-in-out;">
                                </div>
                        </div>
                    </div>
                    """
                details_html += "</div>"
            else:
                overall_status_text = "Status: This appears to be a Good Comment (refined check)." 
                details_html = "<p style='color: green; font-weight: bold;'>No specific toxicities detected at a significant level.</p>"
                
        return overall_status_text, details_html

    except requests.exceptions.RequestException as e:
        overall_status_text = f"API Connection Error: {e}"
        details_html = f"<p style='color: red;'>Could not connect to the analysis backend.</p>"
        return overall_status_text, details_html
    except json.JSONDecodeError:
        overall_status_text = "API Response Error: Invalid JSON format."
        details_html = "<p style='color: red;'>The backend returned an invalid response format.</p>"
        return overall_status_text, details_html
    except Exception as e:
        overall_status_text = f"An Unexpected Error Occurred: {e}"
        details_html = f"<p style='color: red;'>An unexpected error occurred during processing.</p>"
        return overall_status_text, details_html

input_comment = gr.Textbox(lines=5, label="Enter Comment Text Here", placeholder="Type or paste your comment...")
output_status = gr.Textbox(label="Overall Comment Status", interactive=False)
output_details_html = gr.HTML(label="Toxicity Details")

iface = gr.Interface(
    fn=get_toxicity_analysis,
    inputs=input_comment,
    outputs=[output_status, output_details_html],
    title="Toxicity Detector UI ",
    description="Enter a comment to analyze its toxicity. The system will provide an overall status and details if toxicity is detected.",
    examples=[ 
        ["Thank you so much for your help, I really appreciate it!"],
        ["What a beautiful sunny day, perfect for a walk in the park."],
        ["Congratulations on your new achievement, that's fantastic news!"],
        ["You are a complete idiot and your opinion is worthless garbage."],
        ["I'm going to find you and make you pay for this, you pathetic loser."],
        ["This is the most disgusting and vile thing I have ever seen, it makes me sick."],
        ["Well, that was an incredibly stupid mistake, wasn't it?"],
    ],
    allow_flagging="never"
)

if __name__ == "__main__":
    print("Launching Gradio UI...")
    iface.launch()