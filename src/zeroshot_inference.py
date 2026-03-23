# Importar librerías necesarias
import pandas as pd
import numpy as np
import requests
import time
import os

# Configurar conexión a LM Studio
url = "http://192.168.195.1:1234/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}

# Directorio de los subsets
subsets_root = r"C:\Users\USER\Desktop\Productos\datosSOY_SUPER\allowed_labels_subsets"

# Directorio para guardar resultados
output_dir = r"C:\Users\USER\Desktop\TFM_Archivos\resultados_zero_shot_codestral"
os.makedirs(output_dir, exist_ok=True)

# Crear el prompt base
prompt_template = (
    "Choose one product type from the list below.\n"
    "Write only the product type name exactly as shown. Do not add anything else.\n\n"
    "List of product types:\n"
)




# Crear función para construir el texto de entrada
def prepare_text(row):
    return f"{row['DESC_ART']} | {row['CATEGORIA']}"

# Función para consultar al modelo
def get_response(text, allowed_labels):
    labels_text = "\n".join(f"- {label}" for label in allowed_labels)
    full_prompt = prompt_template + labels_text + f"\n\nInput:\n\"{text}\"\n\nProduct Type:"

    payload = {
        "model": "codestral-22b-v0.1",
        "messages": [
            {"role": "user", "content": "Classify the product"},
            {"role": "assistant", "content": full_prompt}
        ]

        ,
        "temperature": 0.0,
        "max_tokens": 20
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        result = response.json()
        prediction = result['choices'][0]['message']['content'].strip()
        return prediction
    except Exception as e:
        print(f"Error: {e}")
        return "Error"

# Listar todos los subsets
subset_files = sorted([f for f in os.listdir(subsets_root) if f.endswith(".csv")])

# Procesar cada subset
for subset_file in subset_files:
    subset_name = subset_file.replace('.csv', '')
    output_file = os.path.join(output_dir, f"{subset_name}_predictions.csv")

    # Verificar si ya fue procesado
    if os.path.exists(output_file):
        print(f"⏭️  Subset {subset_name} ya procesado. Saltando...")
        continue

    print(f"\n🔄 Procesando {subset_file}...")

    # Cargar subset
    subset_path = os.path.join(subsets_root, subset_file)
    subset_df = pd.read_csv(subset_path, encoding="utf-8-sig")

    # Crear campo 'text'
    subset_df["text"] = subset_df.apply(prepare_text, axis=1)

    # Obtener etiquetas permitidas
    allowed_labels = sorted(subset_df['DESC_PROD'].unique())

    # Crear columna para predicciones si no existe
    if 'Prediction' not in subset_df.columns:
        subset_df['Prediction'] = np.nan
    subset_df['Prediction'] = subset_df['Prediction'].astype('object')

    # Clasificar fila por fila (solo si no tiene ya predicción)
    for index, row in subset_df.iterrows():
        if pd.notna(row['Prediction']) and row['Prediction'] != "Error":
            continue  # Saltar fila ya predicha correctamente

        text_input = row['text']
        predicted_desc_prod = get_response(text_input, allowed_labels)
        subset_df.at[index, 'Prediction'] = predicted_desc_prod
        time.sleep(1)

        print(f"    Producto {index + 1}/{len(subset_df)} procesado. Respuesta: {predicted_desc_prod}", flush=True)

        # Guardado parcial (tras cada producto)
        subset_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"✅ Predicciones guardadas en: {output_file}")

print("\n✅ Todos los subsets procesados y evaluados.")
