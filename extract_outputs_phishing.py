import json
import base64
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

NOTEBOOK_PATH = r"notebooks/cic_trap4phish_classification.ipynb" 
OUTPUT_DIR = "outputs"

def extract_notebook_data(notebook_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(notebook_path):
        print(f"Error: Could not find notebook at {notebook_path}")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    expected_image_names = ["phishing_model_comparison.png", "phishing_learning_curves.png"]
    image_counter = 0
    
    print("\n--- Extracting Images ---")
    
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        
        for output in cell.get('outputs', []):
            if output.get('output_type') in ['display_data', 'execute_result']:
                data = output.get('data', {})
                if 'image/png' in data:
                    img_data = data['image/png']
                    if isinstance(img_data, list):
                        img_data = "".join(img_data)
                    
                    img_bytes = base64.b64decode(img_data)
                    
                    if image_counter < len(expected_image_names):
                        img_name = expected_image_names[image_counter]
                    else:
                        img_name = f"phishing_extra_figure_{image_counter + 1}.png"
                        
                    img_path = os.path.join(output_dir, img_name)
                    
                    with open(img_path, 'wb') as img_file:
                        img_file.write(img_bytes)
                    
                    print(f"[OK] Saved: {img_path}")
                    image_counter += 1

            text_content = ""
            if output.get('output_type') == 'stream' and output.get('name') == 'stdout':
                text_content = "".join(output.get('text', []))
            elif 'text/plain' in output.get('data', {}):
                text_plain = output['data']['text/plain']
                text_content = "".join(text_plain) if isinstance(text_plain, list) else text_plain

            if text_content:
                if "Performance" in text_content and "Accuracy:" in text_content:
                    print("\n--- Extracted Performance Metrics ---")
                    print(text_content.strip())
                    
    if image_counter == 0:
        print("\n[WARNING] No images found. Ensure notebook was run offline.")

if __name__ == "__main__":
    extract_notebook_data(NOTEBOOK_PATH, OUTPUT_DIR)
