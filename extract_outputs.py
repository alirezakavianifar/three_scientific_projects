import json
import base64
import os

# --- Configuration ---
# Point this to the .ipynb file you downloaded from Colab
NOTEBOOK_PATH = r"notebooks/mnist_classification.ipynb" 
OUTPUT_DIR = "outputs"

def extract_notebook_data(notebook_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(notebook_path):
        print(f"Error: Could not find notebook at {notebook_path}")
        print("Please ensure you have downloaded your executed notebook from Colab and placed it there.")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Standard names for our specific project plots (in order of execution)
    expected_image_names = ["learning_curves.png", "confusion_matrix.png", "prediction_grid.png"]
    image_counter = 0
    
    print("\n--- Extracting Images ---")
    
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        
        for output in cell.get('outputs', []):
            # 1. Extract Images (from display_data or execute_result)
            if output.get('output_type') in ['display_data', 'execute_result']:
                data = output.get('data', {})
                if 'image/png' in data:
                    img_data = data['image/png']
                    if isinstance(img_data, list):
                        img_data = "".join(img_data)
                    
                    # Convert Base64 back to binary PNG
                    img_bytes = base64.b64decode(img_data)
                    
                    # Determine filename
                    if image_counter < len(expected_image_names):
                        img_name = expected_image_names[image_counter]
                    else:
                        img_name = f"extra_figure_{image_counter + 1}.png"
                        
                    img_path = os.path.join(output_dir, img_name)
                    
                    # Save the image
                    with open(img_path, 'wb') as img_file:
                        img_file.write(img_bytes)
                    
                    print(f"[OK] Saved: {img_path}")
                    image_counter += 1

            # 2. Extract Key Text Results (Stdout or Text Outputs)
            text_content = ""
            if output.get('output_type') == 'stream' and output.get('name') == 'stdout':
                text_content = "".join(output.get('text', []))
            elif 'text/plain' in output.get('data', {}):
                text_plain = output['data']['text/plain']
                text_content = "".join(text_plain) if isinstance(text_plain, list) else text_plain

            # Print out important metrics if they appear in the text
            if text_content:
                if "Test Accuracy:" in text_content:
                    print("\n--- Extracted Accuracy ---")
                    print(text_content.strip())
                elif "precision" in text_content and "recall" in text_content:
                    print("\n--- Extracted Classification Report ---")
                    print(text_content.strip())
                    
    if image_counter == 0:
        print("\n[WARNING] No images found in the notebook. Ensure it was successfully run in Colab before downloading.")

if __name__ == "__main__":
    extract_notebook_data(NOTEBOOK_PATH, OUTPUT_DIR)
    print("\nExtraction complete. You can now compile your LaTeX report!")
